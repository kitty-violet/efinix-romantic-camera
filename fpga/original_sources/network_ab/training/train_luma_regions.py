"""Separated lightness and chroma regions, integer shift/add reconstruction."""
import argparse,gc,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from .train_product_regions import ProductNet
from .train_detail_rgb import Training as Base
from .train_region_classes import class_plan
from .region_detail_target import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .coherent_shared_model import warp

LEVELS=[[0,12,26,41,57,74,91,109,127,145,163,182,200,219,237,255],
        [-96,-48,-24,-8,0,8,24,48],[-64,-32,-16,-6,0,6,16,32]]
SLICES=[slice(0,16),slice(16,24),slice(24,32)]

def components(x):
    r,g,b=x.unbind(1)
    return torch.stack((.25*r+.5*g+.25*b,.5*r-.5*b,-.25*r+.5*g-.25*b),1)

def centers(x):return [x.new_tensor(v)/128-(1 if i==0 else 0) for i,v in enumerate(LEVELS)]
def probabilities(p):return [p[:,s].softmax(1) for s in SLICES]
def labels(t):
    c=components(t)
    return [(c[:,i,None]-v[None,:,None,None]).abs().argmin(1) for i,v in enumerate(centers(t))]

def reconstruct(y,co,cg):return torch.stack((y+co-cg,y+cg,y-co-cg),1).clamp(-1,127/128)
def display(p):return reconstruct(*[v[p[:,s].argmax(1)] for v,s in zip(centers(p),SLICES)])
def expected(p):return reconstruct(*[(q*v[None,:,None,None]).sum(1) for q,v in zip(probabilities(p),centers(p))])

class LumaNet(ProductNet):
    def __init__(self,initialize=False):
        super().__init__(False)
        if initialize:
            old=torch.load(ROOT/'runs'/RUN/'product/fp32_checkpoint_003000.pt',map_location='cpu',weights_only=True)['model']
            self.load_state_dict(old)
        nn.init.normal_(self.conv('rgb').weight,std=.04);nn.init.zeros_(self.conv('rgb').bias)

    def integer_package(self):
        p=super().integer_package();p.update(schema='luma-regions-int8-q17-v1',levels=LEVELS,
            output='C32: Y16/Co8/Cg8 logits; argmax with lower-index ties; R=clip(Y+Co-Cg),G=clip(Y+Cg),B=clip(Y-Co-Cg)')
        return p

def prepare(out):
    out.mkdir(exist_ok=False);base=ROOT/'runs'/RUN/'rgb';(out/'data.json').write_bytes((base/'data.json').read_bytes())
    cfg=dict(seed=2026100425,fp32_steps=3000,qat_steps=1500,crop=320,margin=24,native_interval=50,
        fp32_lr=.0015,qat_lr=.00025,lr_floor_factor=.1,levels=LEVELS,
        changed_output='Y16/Co8/Cg8 categorical, shifts/adds reconstruction; hardware verification pending')
    write_json(out/'config.json',cfg);write_json(out/'graph.json',class_plan())
    paths=[Path(__file__)]+[Path(__file__).with_name(p) for p in ('train_product_regions.py','detail_rgb_net.py','train_detail_rgb.py','region_budget_net.py','train_region_classes.py','coherent_shared_model.py')]
    write_json(out/'binding.json',dict(code={str(p):file_hash(p) for p in paths},config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json')))

class Training(Base):
    def __init__(self,out,mode='fp32'):
        super().__init__(out,mode);self.net=LumaNet(True).cuda().eval();self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg[mode+'_lr'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);p=self.net(x);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        labs=labels(t);ce=sum(F.cross_entropy(cut(p[:,s]),cut(l)) for s,l in zip(SLICES,labs))/3
        color=F.l1_loss(cut(expected(p)),cut(t));prob=probabilities(p);temporal=[];phase=[]
        with torch.no_grad():valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)])
        shifted=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];shiftprob=probabilities(self.net(shifted))
        for pr,l,q in zip(prob,labs,shiftprob):
            with torch.no_grad():same=(l[1:]==warp(l[:1,None].float(),flow)[:,0].round().long()).float()*valid[:,0]
            temporal.append((cut((pr[1:]-warp(pr[:1],flow)).square().sum(1))*cut(same)).sum()/cut(same).sum().clamp_min(1))
            phase.append((q[...,m+sy:-m,m+sx:-m]-pr[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).square().sum(1).mean())
        temporal=sum(temporal)/3;phase=sum(phase)/3;ramp=min(1,(self.step+1)/250)
        loss=ce+.6*color+ramp*(temporal+1.5*phase)
        lr=self.cfg[self.mode+'_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg[self.mode+'_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True)
        self.opt.step();self.step+=1;torch.cuda.synchronize()
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

def preflight(out,mode,initialize):
    tr=Training(out,mode)
    if mode=='qat':tr.start_qat(initialize)
    tr.update(True);tr.update(False);torch.save(tr.state(),out/(mode+'_preflight.pt'));a=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out,mode);tr.restore(torch.load(out/(mode+'_preflight.pt'),map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample']
    write_json(out/(mode+'_preflight.json'),dict(status='PASS',full720_pair_backward=True,batch2=True,restore_max_error=err,macs=class_plan()['macs']))
    print('LUMA_PREFLIGHT_PASS',mode,err,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--mode',choices=['fp32','qat'],default='fp32');p.add_argument('--stop',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--initialize',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100425);out=ROOT/'runs'/RUN/'luma'
    if a.action=='prepare':prepare(out);return
    if a.action=='preflight':preflight(out,a.mode,a.initialize);return
    assert json.loads((out/(a.mode+'_preflight.json')).read_text())['status']=='PASS';tr=Training(out,a.mode)
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
            if tr.step%50==0:print('LUMA_TRAIN',a.mode,tr.step,round(row['loss'],4),round(row['ce'],4),flush=True)
            if tr.step%500==0 or tr.step==stop:torch.save(tr.state(),out/f'{a.mode}_checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('LUMA_SEGMENT_DONE',a.mode,tr.step,flush=True)

if __name__=='__main__':main()
