"""Region labels with independent RGB codebooks: 512 colors in 24 logits.

Eight reserved logits retain existing C32 packing. Final argmax8 per channel
requires a small output reducer; this is explicitly not a B1 weight-only update.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from .detail_rgb_net import DetailRGBNet
from .train_region_classes import class_plan
from .train_detail_rgb import Training as RGBTraining
from .region_detail_target import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .coherent_shared_model import warp

LEVELS=[0,17,44,78,116,159,205,255]


class ProductNet(DetailRGBNet):
    def __init__(self,initialize=True):
        super().__init__();self.nodes=class_plan()['nodes']
        for key in list(self.layers):
            if key.startswith('res2_') or key=='finish__pw':del self.layers[key]
        self.layers['rgb']=nn.Sequential(nn.Conv2d(8,32,1))
        if initialize:
            path=ROOT/'runs/region-mechanisms-20261004/classes32/final/region_classes_fp32.pt'
            old=torch.load(path,map_location='cpu',weights_only=True)['model']
            current=self.state_dict()
            self.load_state_dict({k:old[k] if k in old and not k.startswith('layers.rgb.') else v for k,v in current.items()})
        nn.init.normal_(self.conv('rgb').weight,std=.04);nn.init.zeros_(self.conv('rgb').bias)
        with torch.no_grad():self.conv('rgb').weight[24:].zero_()

    def integer_package(self):
        p=super().integer_package();p.update(plan=class_plan(),schema='product-regions-int8-q17-v1',levels=LEVELS,
            output='C32 groups: R logits8/G logits8/B logits8/reserved8. argmax8 each then shared 8-byte codebook ->RGB')
        return p


def classes(p):return p[:,:24].reshape(p.shape[0],3,8,*p.shape[-2:])
def lab(t):
    centers=t.new_tensor(LEVELS)/128-1
    return (t[:,:,None]-centers[None,None,:,None,None]).abs().argmin(2)
def display(p):
    levels=p.new_tensor(LEVELS)/128-1
    return levels[classes(p).argmax(2)]
def probability(p):return classes(p).softmax(2)
def expected(p):return (probability(p)*(p.new_tensor(LEVELS)/128-1)[None,None,:,None,None]).sum(2)


def prepare(out):
    out.mkdir(exist_ok=False);base=ROOT/'runs'/RUN/'rgb'
    for name in ('data.json',): (out/name).write_bytes((base/name).read_bytes())
    cfg=dict(seed=2026100424,fp32_steps=3000,qat_steps=1500,crop=320,margin=24,native_interval=50,
        fp32_lr=.002,qat_lr=.0003,lr_floor_factor=.1,levels=LEVELS,
        changed_output='argmax8 per R/G/B + fixed codebook, reserved4th group; pending RTL area/timing')
    write_json(out/'config.json',cfg);write_json(out/'graph.json',class_plan())
    paths=[Path(__file__)]+[Path(__file__).with_name(p) for p in ('detail_rgb_net.py','train_detail_rgb.py','region_budget_net.py','train_region_classes.py','coherent_shared_model.py')]
    write_json(out/'binding.json',dict(code={str(p):file_hash(p) for p in paths},config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json')))


class Training(RGBTraining):
    def __init__(self,out,mode='fp32'):
        super().__init__(out,mode);self.net=ProductNet().cuda().eval();self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg[mode+'_lr'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);p=self.net(x);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        logits=classes(p);labels=lab(t);lp=cut(logits).reshape(-1,8,*cut(p).shape[-2:]);lt=cut(labels).reshape(-1,*cut(p).shape[-2:])
        ce=F.cross_entropy(lp,lt);y=expected(p);color=F.l1_loss(cut(y),cut(t))
        prob=probability(p);flat=prob.reshape(2,24,*p.shape[-2:]);prev=warp(flat[:1],flow).reshape(1,3,8,*p.shape[-2:])
        with torch.no_grad():
            valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
            labels_same=(labels[1:]==warp(labels[:1].float(),flow).round().long()).float()*valid
        delta=(prob[1:]-prev).square().sum(2)
        temporal=(cut(delta)*cut(labels_same)).sum()/cut(labels_same).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)])
        shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];q=probability(self.net(shift))
        phase=(q[...,m+sy:-m,m+sx:-m]-prob[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).square().sum(2).mean()
        ramp=min(1,(self.step+1)/250);loss=ce+.4*color+ramp*(1.0*temporal+1.5*phase)
        lr0=self.cfg[self.mode+'_lr'];lr=lr0*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg[self.mode+'_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,mode=self.mode,loss=float(loss.detach()),ce=float(ce.detach()),color=float(color.detach()),temporal=float(temporal.detach()),
            phase=float(phase.detach()),grad=float(norm),native=native,lr=lr,sample=spec,seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'{self.mode}_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();p=display(self.net(x));Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'))
            q=display(self.net(F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:1280]));rows.append(dict(id=r['id'],phase_rgb255=float((q[:,:,32:-32,33:-32]-p[:,:,32:-32,32:-33]).abs().mean()*128)))
        write_json(folder/'scores.json',dict(records=rows,style_accepted=False,not_blind=True))
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(960,894),'#18202a');d=ImageDraw.Draw(board)
        for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
            row=next(r for r in self.data['validation'] if r['id']==key);x,y=xy
            for i,(path,title) in enumerate([(row['source'],'原片'),(ROOT/'runs'/RUN/'detail_a'/(key+'.png'),'细节区域目标'),(folder/(key+'.png'),self.mode+f' {self.step}步')]):
                d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
        board.save(folder/'comparison.png')


def preflight(out):
    tr=Training(out);tr.update(True);tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample']
    write_json(out/'preflight.json',dict(status='PASS',full720_pair_backward=True,batch2=True,restore_max_error=err,macs=class_plan()['macs']))
    print('PRODUCT_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--mode',choices=['fp32','qat'],default='fp32');p.add_argument('--stop',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--initialize',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100424);out=ROOT/'runs'/RUN/'product'
    if a.action=='prepare':prepare(out);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out,a.mode)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:
        assert not (out/(a.mode+'_metrics.jsonl')).exists()
        if a.mode=='qat':assert a.initialize;tr.start_qat(a.initialize)
        tr.evaluate()
    write_json(out/f'{a.mode}_launch_{tr.step:06d}.json',dict(start=tr.step,resume=str(a.resume),initialize=str(a.initialize),binding=tr.binding))
    stop=min(a.stop or tr.cfg[a.mode+'_steps'],tr.cfg[a.mode+'_steps'])
    with (out/(a.mode+'_metrics.jsonl')).open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<stop:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('PRODUCT_TRAIN',a.mode,tr.step,round(row['loss'],4),round(row['ce'],4),flush=True)
            if tr.step%500==0 or tr.step==stop:torch.save(tr.state(),out/f'{a.mode}_checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('PRODUCT_SEGMENT_DONE',a.mode,tr.step,flush=True)

if __name__=='__main__':main()
