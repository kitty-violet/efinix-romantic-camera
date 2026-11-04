"""Continue the shared teacher with texture, chroma and edge-order supervision.

The grayscale discriminator idea is informed by the authors' implementation:
https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/train.py
This is not a reproduction of that cartoonization pipeline. All targets remain
the user's reviewed game frames. Inference uses the existing shared generator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

import cv2
import torch
from torch.nn import functional as F

from . import coherent_shared_train as base
from .coherent_shared_model import temporal_residual_loss
from .cut_wide_model import Discriminator
from .runtime import configure_workspace, file_hash, write_json


TEXTURE_LEVELS=('relu1_2','relu2_2','relu3_3')


def gray(x):
    weights=x.new_tensor([.2126,.7152,.0722]).reshape(1,3,1,1)
    return (x*weights).sum(1,keepdim=True)


def centered_gram(x):
    flat=x.flatten(2)
    flat=flat-flat.mean(2,keepdim=True)
    return torch.bmm(flat,flat.transpose(1,2))/(x.shape[1]*x.shape[2]*x.shape[3])


def box(x,k):
    return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='reflect'),k,stride=1)


def edge_floor(source,output):
    """Prevent disappearance of existing strong edges; no new edge is rewarded."""
    terms=[]
    for src,out in zip(base.differences(gray(source)),base.differences(gray(output))):
        mask=(src.abs()>.08).detach()
        terms.append((F.relu(.55*src.abs()-out.abs())*mask).sum()/(mask.sum()+1))
    return sum(terms)


def flat_residual(source,output):
    """Do not invent small-scale texture on locally smooth source surfaces."""
    lum=gray(source)
    variance=(box(lum.square(),9)-box(lum,9).square()).clamp_min(0)
    mask=torch.exp(-variance.detach()/.001)
    residual=output-source
    return ((residual-box(residual,9)).abs()*mask).sum()/(mask.sum()*3+1e-6)


def chroma(x):
    return x-gray(x)


def gradient_order(source,output):
    """Penalize reversal of visible source edges, without locking their amplitude."""
    terms=[]
    src=F.avg_pool2d(gray(source),2);out=F.avg_pool2d(gray(output),2)
    for ds,do in zip(base.differences(src),base.differences(out)):
        mask=(ds.abs()>.02).detach()
        reversed_edge=F.relu(-ds.sign()*do)
        terms.append((reversed_edge*mask).sum()/(mask.sum()+1))
    return sum(terms)


class TextureTraining(base.Training):
    def __init__(self,config):
        super().__init__(config)
        initial=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
        total=initial['step']+initial['config'].get('prior_training_steps',0)
        if total!=config['prior_training_steps'] or initial['config']['model']!=config['model']:
            raise ValueError('Wrong warm-start model')
        if initial['binding']['manifest_sha256']!=file_hash(config['manifest']):
            raise ValueError('Warm-start data mismatch')
        self.net.load_state_dict(initial['ema']);self.ema.load_state_dict(initial['ema'])
        self.disc.load_state_dict(initial['discriminator'])
        self.anchor=copy.deepcopy(self.ema).eval().requires_grad_(False)
        self.texture_disc=Discriminator(width=24).cuda()
        self.texture_opt=torch.optim.Adam(self.texture_disc.parameters(),
                          lr=config['learning_rate']*.5,betas=(.5,.999))
        rows={k:[] for k in TEXTURE_LEVELS}
        with torch.no_grad():
            for row in self.samples.styles:
                im=cv2.resize(self.samples.style(row),(512,288),interpolation=cv2.INTER_AREA)
                feats=self.vgg.extract(gray(base.tensor(im)).repeat(1,3,1,1))
                for k in rows:rows[k].append((row['style_group'],centered_gram(feats[k]).cpu()))
        self.texture_bank={k:torch.stack([
            torch.stack([v for group,v in entries if group==g]).mean(0)
            for g in sorted(self.samples.style_groups)]).mean(0).cuda()
            for k,entries in rows.items()}

    def update(self):
        started=time.perf_counter();self.net.train()
        a,b,flow,mask,style,spec=self.samples.next()
        outa,outb=self.net(torch.cat((a,b),0)).chunk(2,0)
        self.disc.requires_grad_(True);self.texture_disc.requires_grad_(True)
        self.optd.zero_grad(set_to_none=True);self.texture_opt.zero_grad(set_to_none=True)
        real,fake=self.disc(style),self.disc(outa.detach())
        dr=.5*((real-1).square().mean()+fake.square().mean())
        gr=self.texture_disc(gray(style).repeat(1,3,1,1))
        gf=self.texture_disc(gray(outa.detach()).repeat(1,3,1,1))
        dt=.5*((gr-1).square().mean()+gf.square().mean())
        (dr+dt).backward();self.optd.step();self.texture_opt.step()
        self.disc.requires_grad_(False);self.texture_disc.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            src=self.vgg.extract(a)
            anchor_full=self.anchor(a)
            anchored=F.avg_pool2d(anchor_full,32)
        feats=self.vgg.extract(outa)
        tex=self.vgg.extract(gray(outa).repeat(1,3,1,1))
        terms={
            'content':torch.stack([base.relative_mse(feats[k],src[k]) for k in ('relu2_2','relu3_3')]).mean(),
            'style':torch.stack([base.relative_mse(base.gram(feats[k]),self.bank[k]) for k in base.STYLE_LEVELS]).mean(),
            'structure':sum(F.l1_loss(x,y) for x,y in zip(base.differences(F.avg_pool2d(a,4)),base.differences(F.avg_pool2d(outa,4)))),
            'temporal':temporal_residual_loss(a,b,outa,outb,flow,mask),
            'adversarial':(self.disc(outa)-1).square().mean(),
            'texture_adversarial':(self.texture_disc(gray(outa).repeat(1,3,1,1))-1).square().mean(),
            'texture_style':torch.stack([base.relative_mse(centered_gram(tex[k]),self.texture_bank[k]) for k in TEXTURE_LEVELS]).mean(),
            'residual_tv':sum(d.abs().mean() for d in base.differences(outa-a)),
            'anchor':F.l1_loss(F.avg_pool2d(outa,32),anchored),
            'edge_floor':edge_floor(a,outa),
            'flat_residual':flat_residual(a,outa),
            'chroma_anchor':F.l1_loss(F.avg_pool2d(chroma(outa),4),F.avg_pool2d(chroma(anchor_full),4)),
            'gradient_order':gradient_order(a,outa),
        }
        weights=self.config['loss_weights'];loss=sum(weights[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();shift_value=0.
        if self.step%4==0:
            shifted=torch.roll(a,1,3)
            prediction=self.net(shifted)
            with torch.no_grad():expected=torch.roll(self.net(a),1,3)
            shift=F.l1_loss(prediction[:,:,12:-12,12:-12],expected[:,:,12:-12,12:-12])
            (weights['shift']*shift).backward();shift_value=float(shift.detach())
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),10.,error_if_nonfinite=True)
        if float(grad)<=0:raise FloatingPointError('Zero gradient')
        self.opt.step()
        with torch.no_grad():
            decay=min(.995,(1+self.step)/(10+self.step))
            for target,value in zip(self.ema.parameters(),self.net.parameters()):target.lerp_(value,1-decay)
        self.step+=1
        row={'step':self.step,**{k:float(v.detach()) for k,v in terms.items()},
             'discriminator':float(dr.detach()),'texture_discriminator':float(dt.detach()),
             'shift':shift_value,'gradient':float(grad),'seconds':time.perf_counter()-started,
             'sample':spec,'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2}
        if not all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):raise FloatingPointError('Nonfinite metric')
        return row

    def state(self,binding):
        state=super().state(binding)
        state.update(texture_discriminator=self.texture_disc.state_dict(),
                     texture_optimizer=self.texture_opt.state_dict(),
                     training_variant='guarded-texture-v1')
        return state

    def restore(self,state,binding):
        super().restore(state,binding)
        if state['training_variant']!='guarded-texture-v1':raise ValueError('Wrong trainer state')
        self.texture_disc.load_state_dict(state['texture_discriminator'])
        self.texture_opt.load_state_dict(state['texture_optimizer'])


def binding_for(config,path):
    code=[Path(__file__),Path(base.__file__),Path(__file__).with_name('coherent_shared_model.py'),
          Path(__file__).with_name('losses.py'),Path(__file__).with_name('cut_wide_model.py')]
    return {'code':{str(p):file_hash(p) for p in code},'config_sha256':file_hash(path),
        'manifest_sha256':file_hash(config['manifest']),
        'initial_checkpoint_sha256':file_hash(config['initial_checkpoint']),
        'hardware_contract_sha256':file_hash(r'E:\fpga\network_ab\model_b_contract.json'),
        'role':'experimental_shared_teacher_not_hardware_contract_model'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int,default=1500)
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8'))
    root=configure_workspace()
    if (root/'runs').resolve() not in a.run.resolve().parents:raise ValueError('Wrong run root')
    if (a.run/'status.json').exists() and not a.resume:raise ValueError('Run already exists')
    binding=binding_for(config,a.config);a.run.mkdir(parents=True,exist_ok=True)
    train=TextureTraining(config)
    source=a.run/'source';source.mkdir(exist_ok=True)
    for path in [Path(k) for k in binding['code']]+[a.config]:shutil.copy2(path,source/path.name)
    write_json(a.run/'run.json',{'binding':binding,'config':config,'precision':'FP32-no-AMP-no-TF32'})
    if a.preflight:
        initial=a.run/'stage_initial.pt';torch.save(train.state(binding),initial)
        rows=[train.update() for _ in range(3)]
        saved=a.run/'preflight_resume.pt';torch.save(train.state(binding),saved)
        reference=train.update();expected={k:v.detach().cpu().clone() for k,v in train.net.state_dict().items()}
        train.restore(torch.load(saved,map_location='cpu',weights_only=True),binding)
        replay=train.update()
        error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in train.net.state_dict().items())
        with torch.no_grad():full=train.ema(torch.zeros(1,3,720,1280,device='cuda'))
        if error>1e-5 or full.shape!=(1,3,720,1280) or not torch.isfinite(full).all():raise RuntimeError('GPU preflight failed')
        write_json(a.run/'preflight.json',{'status':'PASS','binding':binding,'steps':rows,
            'resume_reference':reference,'resume_replay':replay,'resume_max_abs_error':error,
            'full_720_shape':list(full.shape),'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2})
        print(json.dumps({'status':'PASS','resume_max_abs_error':error,'stage_initial':str(initial)}),flush=True)
        return
    check=json.loads((Path(config['preflight_run'])/'preflight.json').read_text())
    if check['status']!='PASS' or check['binding']!=binding:raise ValueError('Matching GPU preflight required')
    if a.resume:train.restore(torch.load(a.resume,map_location='cpu',weights_only=True),binding)
    log=a.run/f'metrics_from_{train.step:06d}.jsonl'
    try:
        while train.step<a.stop_after:
            row=train.update()
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            if train.step%10==0 or train.step==1:
                write_json(a.run/'progress.json',{'status':'RUNNING',**row})
                print(json.dumps({k:row[k] for k in ('step','content','style','texture_style','temporal','seconds')}),flush=True)
            if train.step in config['preview_steps']:base.previews(train,a.run)
            if train.step%config['checkpoint_every']==0 or train.step==a.stop_after:
                if binding_for(config,a.config)!=binding:raise ValueError('Code/data/config drift')
                checkpoint=a.run/f'checkpoint_{train.step:06d}.pt';temp=checkpoint.with_suffix('.tmp')
                torch.save(train.state(binding),temp);os.replace(temp,checkpoint)
        write_json(a.run/'status.json',{'status':'STOP_AFTER_REACHED','step':train.step,
            'quality_gate':'UNREVIEWED','checkpoint':str(checkpoint),'sha256':file_hash(checkpoint)})
    except Exception as error:
        write_json(a.run/'status.json',{'status':'FAILED','step':train.step,'error':str(error)})
        raise


if __name__=='__main__':main()
