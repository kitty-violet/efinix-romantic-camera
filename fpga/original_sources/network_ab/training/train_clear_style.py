"""FP32 paired appearance distillation with real motion and resumable state."""
import argparse
import copy
from functools import lru_cache
import gc
import json
from pathlib import Path
import random
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_shared_model import warp, temporal_residual_loss
from .engine import set_reproducible
from .losses import PerceptualLoss
from .runtime import configure_workspace, file_hash, write_json


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1)[None].cuda().float()/128-1


def differences(x):return x[:,:,1:]-x[:,:,:-1],x[:,:,:,1:]-x[:,:,:,:-1]


class Samples:
    def __init__(self,path,seed):
        self.path=Path(path);self.data=json.loads(self.path.read_text(encoding='utf-8'))
        if self.data['status']!='REVIEWED_FOR_BOUNDED_DISTILLATION':raise ValueError('Unreviewed teacher supervision')
        self.rng=random.Random(seed);self.groups={}
        for row in self.data['pairs']:
            if self.data['source_splits'][row['source']]['split']!='train':raise ValueError('Source leakage')
            for k in ['current','next','flow','current_target','next_target']:
                if file_hash(row[k])!=row[k+'_sha256']:raise ValueError('Data drift '+k)
            self.groups.setdefault(row['source'],[]).append(row)

    @lru_cache(maxsize=48)
    def image(self,path):return np.array(Image.open(path).convert('RGB'))

    def next(self):
        rng=self.rng;row=rng.choice(self.groups[rng.choice(sorted(self.groups))])
        images=[self.image(row[k]) for k in ['current','next','current_target','next_target']]
        with np.load(row['flow']) as f:flow=f['backward'].astype(np.float32);mask=f['mask'].astype(np.float32)
        wide=rng.random()<.2
        if wide:
            images=[cv2.resize(v,(640,360),interpolation=cv2.INTER_AREA) for v in images]
            xy=[0,0]
        else:
            flow=cv2.resize(flow,(1280,720))*2;mask=cv2.resize(mask,(1280,720),interpolation=cv2.INTER_NEAREST)
            x,y=rng.randrange(1280-384+1),rng.randrange(720-384+1);xy=[x,y]
            images=[v[y:y+384,x:x+384] for v in images];flow=flow[y:y+384,x:x+384];mask=mask[y:y+384,x:x+384]
        h,w=mask.shape;yy,xx=np.mgrid[:h,:w]
        mask*=((xx+flow[:,:,0]>=4)&(xx+flow[:,:,0]<w-4)&(yy+flow[:,:,1]>=4)&(yy+flow[:,:,1]<h-4))
        flip=rng.random()<.5
        if flip:
            images=[v[:,::-1].copy() for v in images];flow=flow[:,::-1].copy();flow[:,:,0]*=-1;mask=mask[:,::-1].copy()
        flow=torch.from_numpy(np.ascontiguousarray(flow)).permute(2,0,1)[None].cuda()
        mask=torch.from_numpy(np.ascontiguousarray(mask))[None,None].cuda()
        return [tensor(v) for v in images],flow,mask,{'source':row['source'],'parent_pair_index':row['parent_pair_index'],'wide':wide,'xy':xy,'flip':flip}


class Training:
    def __init__(self,root,config):
        self.config=config;self.step=0;torch.manual_seed(config['seed']);random.seed(config['seed'])
        self.samples=Samples(config['manifest'],config['seed'])
        self.net=SurfaceAppearanceNet(**config['model']).cuda()
        self.ema=copy.deepcopy(self.net).eval().requires_grad_(False)
        self.vgg=PerceptualLoss(root).cuda();self.vgg.features=self.vgg.features[:16]
        self.opt=torch.optim.Adam(self.net.parameters(),lr=config['learning_rate'])

    def update(self):
        begin=time.perf_counter();self.net.train()
        (a,b,ta,tb),flow,mask,spec=self.samples.next();x=torch.cat([a,b]);target=torch.cat([ta,tb])
        out=self.net(x);oa,ob=out.chunk(2)
        with torch.no_grad():tf=self.vgg.extract(F.avg_pool2d(target,2))
        of=self.vgg.extract(F.avg_pool2d(out,2))
        perceptual=torch.stack([F.mse_loss(of[k],tf[k])/(tf[k].square().mean()+1e-6) for k in ['relu2_2','relu3_3']]).mean()
        blur=lambda v:F.avg_pool2d(v,5,1,2)
        errors=(ob-tb)-warp(oa-ta,flow)
        terms={'paired':F.l1_loss(out,target),
               'multiscale':F.l1_loss(F.avg_pool2d(out,8),F.avg_pool2d(target,8)),
               'detail':F.l1_loss(out-blur(out),target-blur(target)),
               'edge':sum(F.l1_loss(u,v) for u,v in zip(differences(out),differences(target))),
               'perceptual':perceptual,
               'temporal_error':(errors.abs()*mask).sum()/(mask.sum()*3+1e-6),
               'temporal_style':temporal_residual_loss(a,b,oa,ob,flow,mask)}
        weights=self.config['loss_weights'];loss=sum(v*weights[k] for k,v in terms.items())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        self.opt.zero_grad(set_to_none=True);loss.backward()
        shift_value=0.
        if self.step%4==0:
            shifted=self.net(torch.roll(a,1,3))
            with torch.no_grad():expected=torch.roll(self.net(a),1,3)
            shift=F.l1_loss(shifted[:,:,16:-16,16:-16],expected[:,:,16:-16,16:-16]);(weights['shift']*shift).backward();shift_value=float(shift.detach())
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),10,error_if_nonfinite=True)
        if float(grad)<=0:raise FloatingPointError('Zero gradient')
        self.opt.step();self.step+=1
        with torch.no_grad():
            decay=min(.98,(1+self.step)/(10+self.step))
            for dst,src in zip(self.ema.parameters(),self.net.parameters()):dst.lerp_(src,1-decay)
        torch.cuda.synchronize()
        return {'step':self.step,'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},'shift':shift_value,
                'grad_norm':float(grad),'seconds':time.perf_counter()-begin,'peak_MiB':torch.cuda.max_memory_allocated()/2**20,'sample':spec}

    def state(self,binding):
        return {'schema':'clear-style-fp32-checkpoint-v1','config':self.config,'binding':binding,'step':self.step,
                'generator':self.net.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.opt.state_dict(),
                'sampler_rng':self.samples.rng.getstate(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()}

    def restore(self,state,binding):
        if state['schema']!='clear-style-fp32-checkpoint-v1' or state['config']!=self.config or state['binding']!=binding:raise ValueError('Resume binding drift')
        self.net.load_state_dict(state['generator']);self.ema.load_state_dict(state['ema']);self.opt.load_state_dict(state['optimizer']);self.step=state['step']
        self.samples.rng.setstate(state['sampler_rng']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])


def validate(train,root,out):
    out.mkdir(exist_ok=True);rows=[]
    for case in ['quiet_stairs','plaza_stone','park_people']:
        src=root/'runs/clear-style-control-20261003/anime'/case/'source.png'
        target=Path(train.config['validation_target_directory'])/(case+'.png')
        x=tensor(np.array(Image.open(src)));t=tensor(np.array(Image.open(target)))
        with torch.no_grad():y=train.ema(x)
        arr=(y[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
        dst=out/(case+'.png');Image.fromarray(arr).save(dst)
        rows.append({'case':case,'teacher_mae_rgb255':float((y-t).abs().mean()*128),'source_mae_rgb255':float((y-x).abs().mean()*128),'path':str(dst),'sha256':file_hash(dst)})
    write_json(out/'report.json',{'step':train.step,'records':rows,'note':'Previously viewed regression, not blind generalization or aesthetic score'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','train']);ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=2000);args=ap.parse_args()
    root=configure_workspace();run=root/'runs/clear-style-control-20261003';config=json.loads((run/'training_config.json').read_text(encoding='utf-8'))
    set_reproducible(config['seed']);cv2.setNumThreads(2)
    code=[Path(__file__),Path(__file__).with_name('coherent_shared_model.py'),Path(__file__).with_name('coherent_surface_model.py'),Path(__file__).with_name('losses.py')]
    binding={'manifest_sha256':file_hash(config['manifest']),'config_sha256':file_hash(run/'training_config.json'),'code':{str(p):file_hash(p) for p in code},'dtype':'FP32','tf32':False}
    t=Training(root,config)
    if args.mode=='preflight':
        out=run/'preflight';out.mkdir(exist_ok=True)
        x=tensor(np.array(Image.open(root/'runs/clear-style-control-20261003/anime/quiet_stairs/source.png')))
        with torch.no_grad():y=t.net(x)
        identity=float((y-x).abs().max());assert identity==0
        rows=[t.update() for _ in range(3)];saved=out/'resume.pt';torch.save(t.state(binding),saved)
        ref=t.update();expected={k:v.detach().cpu().clone() for k,v in t.net.state_dict().items()}
        del t,x,y;gc.collect();torch.cuda.empty_cache()
        t=Training(root,config);t.restore(torch.load(saved,map_location='cpu',weights_only=True),binding);replay=t.update()
        err=max((v.detach().cpu()-expected[k]).abs().max().item() for k,v in t.net.state_dict().items())
        assert ref['sample']==replay['sample'] and err<=2e-6
        assert all(p.dtype==torch.float32 for p in t.net.parameters())
        write_json(out/'report.json',{'status':'PASS','initial_native720_identity_max':identity,'steps':rows,'reference':ref,'replay':replay,'resume_max_parameter_error':err,'tolerance':2e-6,'parameters':sum(p.numel() for p in t.net.parameters()),'dtype':'FP32','binding':binding})
        print('PREFLIGHT_PASS',err,'last_step_seconds',replay['seconds'],flush=True);return
    pre=json.loads((run/'preflight/report.json').read_text(encoding='utf-8'));assert pre['status']=='PASS' and pre['binding']==binding
    out=run/'student_train';out.mkdir(exist_ok=True)
    if (out/'progress.json').exists() and not args.resume:raise FileExistsError('Existing training; resume explicitly')
    if args.resume:t.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
    sources=out/'source';sources.mkdir(exist_ok=True)
    for p in code:shutil.copy2(p,sources/p.name)
    shutil.copy2(run/'training_config.json',sources/'training_config.json')
    write_json(out/'run.json',{'config':config,'binding':binding,'status':'RUNNING','warm_start':'fresh identity initialization, no old-game weights'})
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while t.step<min(args.stop_after,config['max_steps']):
            row=t.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if t.step%25==0:write_json(out/'progress.json',row);print(json.dumps({k:row[k] for k in ['step','loss','paired','temporal_style','seconds','peak_MiB']}),flush=True)
            if t.step%500==0:
                torch.save(t.state(binding),out/f'checkpoint_{t.step:06d}.pt');validate(t,root,out/f'validation_{t.step:06d}')
        if t.step%500:torch.save(t.state(binding),out/f'checkpoint_{t.step:06d}.pt')
    write_json(out/'status.json',{'status':'COMPLETED_BOUNDED_TRAINING','step':t.step,'requires_video_review':True})


if __name__=='__main__':main()
