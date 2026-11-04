"""FP32 multi-scene appearance training with real video temporal supervision.

Uses existing local VGG19 weights, a shared normalization-free generator, and
training-only PatchGAN/optical flow. No per-video inference fitting.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .coherent_shared_model import SharedAppearanceNet, temporal_residual_loss
from .cut_wide_model import Discriminator
from .engine import set_reproducible
from .losses import PerceptualLoss, gram
from .runtime import configure_workspace, file_hash, write_json


STYLE_LEVELS=('relu1_2','relu2_2','relu3_3','relu4_3')


def tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().unsqueeze(0).cuda()/128-1


class Samples:
    def __init__(self,manifest,seed):
        self.data=json.loads(Path(manifest).read_text(encoding='utf-8'))
        if self.data['status']!='REVIEWED_SHARED_TRAINING':raise ValueError('Unreviewed data')
        self.rng=random.Random(seed); self.cache={}; self.groups={}
        content_sources={r['source']:r['split'] for r in self.data['content']}
        for p in self.data['pairs']:
            if content_sources[p['source']]!='train':raise ValueError('Holdout leakage')
            for k in ('current','next','flow'):
                if file_hash(p[k])!=p[k+'_sha256']:raise ValueError('Temporal data drift')
            self.groups.setdefault(p['source'],[]).append(p)
        self.styles=self.data['styles']
        for s in self.styles:
            if file_hash(s['path'])!=s['sha256']:raise ValueError('Target drift')
        self.style_groups={k:[r for r in self.styles if r['style_group']==k]
                           for k in sorted({r['style_group'] for r in self.styles})}

    def image(self,path):
        if path not in self.cache:self.cache[path]=np.array(Image.open(path).convert('RGB'))
        return self.cache[path]

    def style(self,row):
        image=self.image(row['path']); x1,y1,x2,y2=row['safe_roi_xyxy']
        return image[y1:y2,x1:x2]

    def next(self):
        rng=self.rng
        p=rng.choice(self.groups[rng.choice(sorted(self.groups))])
        a,b=self.image(p['current']),self.image(p['next'])
        with np.load(p['flow']) as flowfile:
            flow=flowfile['backward'].astype(np.float32)
            mask=flowfile['mask'].astype(np.float32)
        wide=rng.random()<.35
        if wide:
            a=cv2.resize(a,(640,360),interpolation=cv2.INTER_AREA)
            b=cv2.resize(b,(640,360),interpolation=cv2.INTER_AREA)
        else:
            flow=cv2.resize(flow,(1280,720),interpolation=cv2.INTER_LINEAR)*2
            mask=cv2.resize(mask,(1280,720),interpolation=cv2.INTER_NEAREST)
            x,y=rng.randrange(1280-384+1),rng.randrange(720-384+1)
            a,b=a[y:y+384,x:x+384],b[y:y+384,x:x+384]
            flow,mask=flow[y:y+384,x:x+384],mask[y:y+384,x:x+384]
        h,w=a.shape[:2]
        yy,xx=np.mgrid[:h,:w]
        mask*=((xx+flow[:,:,0]>=3)&(xx+flow[:,:,0]<w-3)&
               (yy+flow[:,:,1]>=3)&(yy+flow[:,:,1]<h-3))
        if rng.random()<.5:
            a,b=a[:,::-1].copy(),b[:,::-1].copy()
            flow,mask=flow[:,::-1].copy(),mask[:,::-1].copy();flow[:,:,0]*=-1
        s=rng.choice(self.style_groups[rng.choice(sorted(self.style_groups))])
        style=self.style(s)
        if wide:
            style=cv2.resize(style,(w,h),interpolation=cv2.INTER_AREA)
        else:
            sy,sx=style.shape[:2]
            if min(sy,sx)<384:raise ValueError('Style crop too small')
            x,y=rng.randrange(sx-384+1),rng.randrange(sy-384+1)
            style=style[y:y+384,x:x+384]
        flow=torch.from_numpy(np.ascontiguousarray(flow)).permute(2,0,1).unsqueeze(0).cuda()
        mask=torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).unsqueeze(0).cuda()
        return tensor(a),tensor(b),flow,mask,tensor(style),{
            'source':p['source'],'current_index':p['current_index'],
            'next_index':p['next_index'],'target':s['path'],'wide':wide,
            'valid_fraction':float(mask.mean()),'shape':[h,w]}


def relative_mse(a,b):
    return F.mse_loss(a,b)/(b.square().mean().detach()+1e-8)


def differences(x):
    return x[:,:,1:]-x[:,:,:-1],x[:,:,:,1:]-x[:,:,:,:-1]


class Training:
    def __init__(self,config):
        root=configure_workspace();set_reproducible(config['seed'])
        self.config=config;self.samples=Samples(config['manifest'],config['seed'])
        self.net=SharedAppearanceNet(**config['model']).cuda()
        self.ema=copy.deepcopy(self.net).eval().requires_grad_(False)
        self.disc=Discriminator(width=32).cuda()
        self.vgg=PerceptualLoss(root).cuda()
        self.opt=torch.optim.Adam(self.net.parameters(),lr=config['learning_rate'],betas=(.9,.999))
        self.optd=torch.optim.Adam(self.disc.parameters(),lr=config['learning_rate']*.5,betas=(.5,.999))
        self.step=0
        bank={k:[] for k in STYLE_LEVELS}
        with torch.no_grad():
            for row in self.samples.styles:
                im=self.samples.style(row)
                # One common target bank, with per-group balancing, fitted only to game images.
                im=cv2.resize(im,(512,288),interpolation=cv2.INTER_AREA)
                features=self.vgg.extract(tensor(im))
                for k in bank:bank[k].append((row['style_group'],gram(features[k]).cpu()))
        self.bank={}
        for k,rows in bank.items():
            means=[torch.stack([v for group,v in rows if group==g]).mean(0)
                   for g in sorted(self.samples.style_groups)]
            self.bank[k]=torch.stack(means).mean(0).cuda()

    def update(self):
        started=time.perf_counter();self.net.train()
        a,b,flow,mask,style,spec=self.samples.next()
        outa,outb=self.net(torch.cat((a,b),0)).chunk(2,0)
        self.disc.requires_grad_(True);self.optd.zero_grad(set_to_none=True)
        pr,pf=self.disc(style),self.disc(outa.detach())
        dl=.5*(F.mse_loss(pr,torch.ones_like(pr))+F.mse_loss(pf,torch.zeros_like(pf)))
        dl.backward();self.optd.step();self.disc.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():src=self.vgg.extract(a)
        feats=self.vgg.extract(outa)
        content=torch.stack([relative_mse(feats[k],src[k]) for k in ('relu2_2','relu3_3')]).mean()
        style_loss=torch.stack([relative_mse(gram(feats[k]),self.bank[k]) for k in STYLE_LEVELS]).mean()
        af,bf=F.avg_pool2d(a,4),F.avg_pool2d(outa,4)
        structural=sum(F.l1_loss(x,y) for x,y in zip(differences(af),differences(bf)))
        temporal=temporal_residual_loss(a,b,outa,outb,flow,mask)
        pred=self.disc(outa);adv=F.mse_loss(pred,torch.ones_like(pred))
        residual_tv=sum(d.abs().mean() for d in differences(outa-a))
        weights=self.config['loss_weights']
        loss=(weights['content']*content+weights['style']*style_loss+
              weights['structure']*structural+weights['temporal']*temporal+
              weights['adversarial']*adv+weights['residual_tv']*residual_tv)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        shift_value=0.
        # Independent of real-flow accuracy: a one-pixel input shift must not change the look.
        if self.step%4==0:
            shifted=torch.roll(a,shifts=1,dims=3)
            prediction=self.net(shifted)
            with torch.no_grad():expected=torch.roll(self.net(a),shifts=1,dims=3)
            shift=F.l1_loss(prediction[:,:,12:-12,12:-12],expected[:,:,12:-12,12:-12])
            (weights['shift']*shift).backward();shift_value=float(shift)
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),10.,error_if_nonfinite=True)
        if float(grad)<=0:raise FloatingPointError('Zero gradient')
        self.opt.step()
        with torch.no_grad():
            decay=min(.995,(1+self.step)/(10+self.step))
            for target,value in zip(self.ema.parameters(),self.net.parameters()):
                target.lerp_(value,1-decay)
        self.step+=1
        row={'step':self.step,'content':float(content),'style':float(style_loss),
             'structure':float(structural),'temporal':float(temporal),'adv':float(adv),
             'discriminator':float(dl),'residual_tv':float(residual_tv),'shift':shift_value,
             'gradient':float(grad),'seconds':time.perf_counter()-started,'sample':spec,
             'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2}
        if not all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
            raise FloatingPointError('Nonfinite metric')
        return row

    def state(self,binding):
        return {'schema':'coherent-shared-checkpoint-v1','step':self.step,'config':self.config,
                'binding':binding,'generator':self.net.state_dict(),'ema':self.ema.state_dict(),
                'discriminator':self.disc.state_dict(),'optimizer':self.opt.state_dict(),
                'optimizer_d':self.optd.state_dict(),'sampler_rng':self.samples.rng.getstate(),
                'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()}

    def restore(self,state,binding):
        if state['config']!=self.config or state['binding']!=binding:raise ValueError('Resume drift')
        self.net.load_state_dict(state['generator']);self.ema.load_state_dict(state['ema'])
        self.disc.load_state_dict(state['discriminator']);self.opt.load_state_dict(state['optimizer'])
        self.optd.load_state_dict(state['optimizer_d']);self.samples.rng.setstate(state['sampler_rng'])
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);self.step=state['step']


@torch.no_grad()
def previews(training,run):
    out=run/f'validation_{training.step:06d}';out.mkdir(exist_ok=True)
    sources={}
    for row in training.samples.data['content']:
        if row['split']=='val':sources.setdefault(row['source'],[]).append(row)
    records=[]
    # Validation only; the extra indoor holdout and test are not used for checkpoint selection.
    for source,rows in sorted(sources.items()):
        row=rows[len(rows)//2];im=training.samples.image(row['path'])
        output=training.ema(tensor(im))
        rgb=(output[0].cpu().permute(1,2,0).numpy()*128+128).round().clip(0,255).astype(np.uint8)
        path=out/f'{Path(source).stem}.jpg'
        panel=np.concatenate((im,rgb),axis=1)
        Image.fromarray(panel).save(path,quality=95)
        records.append({'source':row['path'],'output':str(path)})
    write_json(out/'review.json',{'status':'VISUAL_REVIEW_REQUIRED','records':records})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stop-after',type=int,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    root=configure_workspace()
    if (root/'runs').resolve() not in args.run.resolve().parents:raise ValueError('Wrong run root')
    code=[Path(__file__),Path(__file__).with_name('coherent_shared_model.py'),
          Path(__file__).with_name('losses.py'),Path(__file__).with_name('cut_wide_model.py')]
    contract=Path(r'E:\fpga\network_ab\model_b_contract.json')
    binding={'code':{str(p):file_hash(p) for p in code},'config_sha256':file_hash(args.config),
             'manifest_sha256':file_hash(config['manifest']),'hardware_contract_sha256':file_hash(contract),
             'role':'experimental_shared_teacher_not_hardware_contract_model'}
    args.run.mkdir(parents=True,exist_ok=True)
    training=Training(config)
    source=args.run/'source';source.mkdir(exist_ok=True)
    for p in code+[args.config]:shutil.copy2(p,source/p.name)
    write_json(args.run/'run.json',{'binding':binding,'config':config,'precision':'FP32-no-AMP-no-TF32',
               'parameters':sum(p.numel() for p in training.net.parameters()),
               'training_sources':sorted(training.samples.groups),'style_count':len(training.samples.styles)})
    if args.preflight:
        rows=[training.update() for _ in range(3)]
        path=args.run/'preflight_resume.pt';torch.save(training.state(binding),path)
        reference=training.update();expected={k:v.detach().cpu().clone() for k,v in training.net.state_dict().items()}
        training.restore(torch.load(path,map_location='cpu',weights_only=True),binding)
        replay=training.update()
        error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in training.net.state_dict().items())
        with torch.no_grad():full=training.ema(torch.zeros(1,3,720,1280,device='cuda'))
        if error>1e-5 or full.shape!=(1,3,720,1280) or not torch.isfinite(full).all():
            raise RuntimeError(f'GPU preflight failed: {error}')
        result={'status':'PASS','binding':binding,'steps':rows,'resume_reference':reference,
                'resume_replay':replay,'resume_max_abs_error':error,'full_720_shape':list(full.shape),
                'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2}
        write_json(args.run/'preflight.json',result);print(json.dumps(result,ensure_ascii=False))
        return
    preflight=json.loads((Path(config['preflight_run'])/'preflight.json').read_text(encoding='utf-8'))
    if preflight['status']!='PASS' or preflight['binding']!=binding:raise ValueError('Missing matching GPU preflight')
    if args.resume:training.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
    elif (args.run/'progress.json').exists():raise ValueError('Run already exists; resume explicitly')
    log=args.run/f'metrics_from_{training.step:06d}.jsonl'
    try:
        while training.step<args.stop_after:
            row=training.update()
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if training.step%10==0 or training.step==1:
                write_json(args.run/'progress.json',{'status':'RUNNING',**row})
                print(json.dumps({k:row[k] for k in ('step','content','style','temporal','seconds') }),flush=True)
            if training.step in config['preview_steps']:previews(training,args.run)
            if training.step%config['checkpoint_every']==0 or training.step==args.stop_after:
                if file_hash(contract)!=binding['hardware_contract_sha256']:raise ValueError('Shared contract changed')
                for p in code:
                    if file_hash(p)!=binding['code'][str(p)]:raise ValueError('Training code changed')
                path=args.run/f'checkpoint_{training.step:06d}.pt';temp=path.with_suffix('.tmp')
                torch.save(training.state(binding),temp);os.replace(temp,path)
        write_json(args.run/'status.json',{'status':'STOP_AFTER_REACHED','step':training.step,
                    'quality_gate':'UNREVIEWED','checkpoint':str(path),'sha256':file_hash(path)})
    except Exception as error:
        write_json(args.run/'status.json',{'status':'FAILED','step':training.step,'error':str(error)})
        raise


if __name__=='__main__':main()
