"""Actual adjacent-frame learning of a finer region target, then exact-Q17 QAT."""
import argparse
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from .detail_rgb_net import DetailRGBNet, plan
from . import detail_rgb_net
from . import region_detail_target as targets
from .region_detail_target import ROOT,RUN,manifest,target
from .region_mechanism_probe import tensor,rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_distillation import as_tensor,to_rgb,read_image,read_flow,edges
from .coherent_shared_model import warp


def prepare(out):
    out.mkdir(exist_ok=False);folder=out/'targets';folder.mkdir();old=manifest();records=[]
    for i,row in enumerate(old['pairs']):
        assert old['source_splits'][row['source']]['split']=='train'
        q=dict(row)
        for k in ('current','next'):
            assert file_hash(row[k])==row[k+'_sha256']
            path=folder/f'{i:03d}_{k}.png'
            Image.fromarray(rgb(target(tensor(read_image(row[k])),'detail_a'))).save(path)
            q[k+'_target']=str(path);q[k+'_target_sha256']=file_hash(path)
        assert row['next_index']==row['current_index']+1
        assert file_hash(row['flow'])==row['flow_sha256']
        records.append(q)
        if (i+1)%10==0:print('RGB_DETAIL_TARGET',i+1,len(old['pairs']),flush=True)
    cfg=dict(seed=2026100421,fp32_steps=4000,qat_steps=2000,crop=320,margin=24,native_interval=50,
        fp32_lr=.002,qat_lr=.00025,lr_floor_factor=.1,target='detail_a',precision='FP32, no AMP/TF32/EMA',
        inference='existing conv/DW/PW/nearest/residual/ReLU and RGB int8 output; changed program graph',
        loss_weights=dict(rgb=2.,coarse=1.,edge=3.,sharp=.8,flat=.4,temporal=.5,phase=.6))
    write_json(out/'config.json',cfg);write_json(out/'graph.json',plan())
    write_json(out/'data.json',dict(pairs=records,validation=old['validation'],source_splits=old['source_splits'],
        training_sources=sorted({r['source'] for r in records}),all_gap1=True,target_kind='detail_a'))
    paths=[Path(__file__),Path(detail_rgb_net.__file__),Path(targets.__file__)]+[Path(__file__).with_name(v) for v in (
        'region_budget_net.py','region_mechanism_probe.py','runtime.py','engine.py','train_b1_distillation.py','coherent_shared_model.py')]
    write_json(out/'binding.json',dict(code={str(p):file_hash(p) for p in paths},
        config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json')))
    print('RGB_DETAIL_PREPARED',len(records),flush=True)


class Training:
    def __init__(self,out,mode='fp32'):
        self.out=out;self.mode=mode;self.cfg=json.loads((out/'config.json').read_text());self.data=json.loads((out/'data.json').read_text(encoding='utf-8'))
        self.binding=json.loads((out/'binding.json').read_text());assert all(file_hash(p)==h for p,h in self.binding['code'].items())
        self.net=DetailRGBNet().cuda().eval();self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg[mode+'_lr']);self.rng=random.Random(self.cfg['seed'])
        self.step=0;self.cache=OrderedDict();self.groups={};self.calibration=None
        for i,row in enumerate(self.data['pairs']):self.groups.setdefault(row['source'],[]).append(i)

    def cached(self,index):
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        r=self.data['pairs'][index]
        images=[read_image(r[k]) for k in ('current','next','current_target','next_target')]
        flow,mask=read_flow(r['flow']);value=(images,flow,mask)
        self.cache[index]=value
        if len(self.cache)>12:self.cache.popitem(last=False)
        return value

    def batch(self,native):
        i=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))]);imgs,flow,mask=self.cached(i)
        w,h=(1280,720) if native else (self.cfg['crop'],)*2
        x=self.rng.randrange(1280-w+1);y=self.rng.randrange(720-h+1);m=self.cfg['margin']
        imgs=torch.stack([as_tensor(v[y:y+h,x:x+w]) for v in imgs]).cuda()
        f=flow[y:y+h,x:x+w].copy();valid=mask[y:y+h,x:x+w].copy();yy,xx=np.mgrid[:h,:w]
        valid*=((xx+f[:,:,0]>=m)&(xx+f[:,:,0]<w-m)&(yy+f[:,:,1]>=m)&(yy+f[:,:,1]<h-m))
        return imgs[:2],imgs[2:],torch.from_numpy(f).permute(2,0,1)[None].cuda(),torch.from_numpy(valid)[None,None].cuda(),[i,x,y,w,h]

    def spatial(self,p,t):
        m=self.cfg['margin'];p,t=[a[:,:,m:-m,m:-m] for a in (p,t)]
        edge=[];sharp=[];flat=[]
        for size in (1,2,4):
            a,b=[F.avg_pool2d(v,size) if size>1 else v for v in (p,t)]
            for gp,gt in zip(edges(a),edges(b)):
                edge.append((gp-gt).abs().mean())
                e=(gt.abs()>.035).float();f=(gt.abs()<.004).float()
                sharp.append((F.relu(gt.abs()-gp.abs()-.003)*e).sum()/e.sum().clamp_min(1))
                flat.append((F.relu(gp.abs()-gt.abs()-.003)*f).sum()/f.sum().clamp_min(1))
        return dict(rgb=F.l1_loss(p,t),coarse=F.l1_loss(F.avg_pool2d(p,8),F.avg_pool2d(t,8)),
            edge=torch.stack(edge).mean(),sharp=torch.stack(sharp).mean(),flat=torch.stack(flat).mean())

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);p=self.net(x);losses=self.spatial(p,t);m=self.cfg['margin'];cut=lambda a:a[:,:,m:-m,m:-m]
        with torch.no_grad():
            valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
            td=(t[1:]-x[1:])-warp(t[:1]-x[:1],flow)
        delta=(p[1:]-x[1:])-warp(p[:1]-x[:1],flow)
        losses['temporal']=(cut(delta-.5*td).abs()*cut(valid)).sum()/(3*cut(valid).sum()+1e-6)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)])
        shifted=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]]
        q=self.net(shifted)
        losses['phase']=F.l1_loss(q[:,:,m+sy:-m,m+sx:-m],p[:1,:,m:-m-sy if sy else -m,m:-m-sx if sx else -m])
        ramp=min(1,(self.step+1)/250)
        loss=sum(v*self.cfg['loss_weights'][k]*(ramp if k in ('phase','temporal') else 1) for k,v in losses.items())
        lr0=self.cfg[self.mode+'_lr'];floor=lr0*self.cfg['lr_floor_factor'];total=self.cfg[self.mode+'_steps']
        lr=floor+(lr0-floor)*.5*(1+math.cos(math.pi*self.step/total))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,mode=self.mode,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},
            grad=float(norm),lr=lr,native=native,sample=spec,seconds=time.perf_counter()-start)

    def state(self):return dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,mode=self.mode,
        sampler=self.rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        scales=self.net.scales,calibration=self.calibration,binding=self.binding,config=self.cfg)

    def restore(self,state):
        assert state['binding']==self.binding and state['config']==self.cfg and state['mode']==self.mode
        self.net.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer']);self.net.scales=state['scales'];self.calibration=state['calibration']
        self.step=state['step'];self.rng.setstate(state['sampler']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])

    def start_qat(self,initial):
        s=torch.load(initial,map_location='cpu',weights_only=True);assert s['binding']==self.binding and s['mode']=='fp32'
        self.net.load_state_dict(s['model']);indices=[]
        for group in self.groups.values():indices.extend([group[0],group[len(group)//2],group[-1]])
        self.calibration=self.net.calibrate(as_tensor(self.cached(i)[0][0])[None].cuda() for i in indices)
        self.calibration.update(indices=indices,initial=str(initial),initial_sha256=file_hash(initial))

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'{self.mode}_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();p=self.net(x);image=to_rgb(p[0]);Image.fromarray(image).save(folder/(r['id']+'.png'))
            teacher=as_tensor(read_image(ROOT/'runs'/RUN/'detail_a'/(r['id']+'.png')))[None].cuda()
            q=self.net(F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:1280])
            rows.append(dict(id=r['id'],target_mae_rgb255=float((p-teacher).abs().mean()*128),
                phase_rgb255=float((q[:,:,32:-32,33:-32]-p[:,:,32:-32,32:-33]).abs().mean()*128)))
        write_json(folder/'scores.json',dict(records=rows,style_accepted=False,not_blind=True))
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(960,894),'#18202a');d=ImageDraw.Draw(board)
        for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
            row=next(r for r in self.data['validation'] if r['id']==key);x,y=xy
            for i,(path,name) in enumerate([(row['source'],'原片'),(ROOT/'runs'/RUN/'detail_a'/(key+'.png'),'本轮区域目标'),(folder/(key+'.png'),self.mode+f' {self.step}步')]):
                d.text((320*i+5,298*j+5),name,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
        board.save(folder/'comparison.png')


def preflight(out):
    tr=Training(out);calls=[]
    hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in tr.net.modules() if isinstance(m,torch.nn.Conv2d)]
    with torch.no_grad():tr.net(torch.zeros(1,3,720,1280,device='cuda'))
    for h in hooks:h.remove()
    assert sum(calls)==plan()['macs']<=966758400
    rows=[tr.update(True),tr.update(False)];torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample']
    write_json(out/'preflight.json',dict(status='PASS',macs=sum(calls),convolutions=len(calls),parameters=sum(p.numel() for p in tr.net.parameters()),
        gpu_full720_pair_backward=True,batch2=True,restore_max_error=err,checks=rows))
    print('DETAIL_RGB_PREFLIGHT_PASS',sum(calls),err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--mode',choices=['fp32','qat'],default='fp32')
    p.add_argument('--stop',type=int);p.add_argument('--resume',type=Path);p.add_argument('--initialize',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100421);cv2.setNumThreads(2);out=ROOT/'runs'/RUN/'rgb'
    if a.action=='prepare':prepare(out);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS'
    tr=Training(out,a.mode)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:
        assert not (out/(a.mode+'_metrics.jsonl')).exists()
        if a.mode=='qat':assert a.initialize;tr.start_qat(a.initialize)
        tr.evaluate()
    start=tr.step;write_json(out/f'{a.mode}_launch_{start:06d}.json',dict(start=start,resume=str(a.resume),initial=str(a.initialize),binding=tr.binding))
    stop=min(a.stop or tr.cfg[a.mode+'_steps'],tr.cfg[a.mode+'_steps'])
    with (out/(a.mode+'_metrics.jsonl')).open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<stop:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('DETAIL_RGB',a.mode,tr.step,round(row['loss'],4),round(row['rgb']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==stop:
                torch.save(tr.state(),out/f'{a.mode}_checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('DETAIL_RGB_SEGMENT_DONE',a.mode,tr.step,flush=True)

if __name__=='__main__':main()
