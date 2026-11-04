"""FP32 controlled repair/distillation: light, texture, and combined ablations."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import torch
from torch.nn import functional as F
from . import coherent_shared_train as base
from .coherent_shared_model import SharedAppearanceNet,warp
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_surface_prior import repair_target,luminance
from .engine import set_reproducible
from .runtime import configure_workspace,file_hash,write_json


def smooth(x,k=5):return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='reflect'),k,stride=1)


class SurfaceTraining:
    def __init__(self,config):
        configure_workspace();set_reproducible(config['seed']);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.config=config;self.samples=base.Samples(config['manifest'],config['seed'])
        self.prior=json.loads(Path(config['prior']).read_text(encoding='utf-8'))
        state=torch.load(config['initial_weights'],map_location='cpu',weights_only=True)
        if state['training_steps']!=config['prior_training_steps'] or state['config']!=config['model']:
            raise ValueError('Wrong warm start')
        if state['binding']['manifest_sha256']!=file_hash(config['manifest']):raise ValueError('Split changed')
        if self.prior['weights_sha256']!=file_hash(config['initial_weights']):raise ValueError('Prior mismatch')
        self.anchor=SharedAppearanceNet(**config['model']).cuda().eval().requires_grad_(False)
        self.anchor.load_state_dict(state['generator'])
        self.net=SurfaceAppearanceNet(**config['model']).cuda();self.net.load_previous(state['generator'])
        self.ema=copy.deepcopy(self.net).eval().requires_grad_(False)
        self.opt=torch.optim.Adam(self.net.parameters(),lr=config['learning_rate'],betas=(.9,.999))
        self.step=0

    def update(self):
        started=time.perf_counter();self.net.train()
        a,b,flow,mask,_,spec=self.samples.next();inputs=torch.cat((a,b),0)
        with torch.no_grad():
            old=self.anchor(inputs)
            target=repair_target(inputs,old,self.prior,self.config['mode'],self.config['surface_strength'])
        output=self.net(inputs);outa,outb=output.chunk(2);ta,tb=target.chunk(2)
        residual=(outb-tb)-warp(outa-ta,flow)
        lum=luminance(output);tl=luminance(target)
        terms={
          'target':F.l1_loss(output,target),
          'tone':F.l1_loss(F.avg_pool2d(output,16),F.avg_pool2d(target,16)),
          'detail':F.l1_loss(lum-smooth(lum),tl-smooth(tl)),
          'edge':sum(F.l1_loss(x,y) for x,y in zip(base.differences(smooth(lum,3)),base.differences(smooth(tl,3)))),
          'temporal':(residual.abs()*mask).sum()/(mask.sum()*3+1e-6),
        }
        weights=self.config['loss_weights'];loss=sum(weights[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        self.opt.zero_grad(set_to_none=True);loss.backward();shift_value=0.
        if self.step%4==0:
            shifted=self.net(torch.roll(a,1,3))
            with torch.no_grad():expected=torch.roll(self.net(a),1,3)
            shift=F.l1_loss(shifted[:,:,12:-12,12:-12],expected[:,:,12:-12,12:-12])
            (weights['shift']*shift).backward();shift_value=float(shift.detach())
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),10.,error_if_nonfinite=True)
        if float(grad)<=0:raise FloatingPointError('Zero gradient')
        self.opt.step()
        with torch.no_grad():
            decay=min(.98,(1+self.step)/(10+self.step))
            for dst,src in zip(self.ema.parameters(),self.net.parameters()):dst.lerp_(src,1-decay)
        self.step+=1
        return {'step':self.step,'mode':self.config['mode'],**{k:float(v.detach()) for k,v in terms.items()},
          'shift':shift_value,'gradient':float(grad),'seconds':time.perf_counter()-started,
          'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2,'sample':spec}

    def state(self,binding):
        return {'schema':'coherent-surface-checkpoint-v1','step':self.step,'config':self.config,
          'binding':binding,'generator':self.net.state_dict(),'ema':self.ema.state_dict(),
          'optimizer':self.opt.state_dict(),'sampler_rng':self.samples.rng.getstate(),
          'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}

    def restore(self,state,binding):
        if state['schema']!='coherent-surface-checkpoint-v1' or state['config']!=self.config or state['binding']!=binding:
            raise ValueError('Resume drift')
        self.net.load_state_dict(state['generator']);self.ema.load_state_dict(state['ema'])
        self.opt.load_state_dict(state['optimizer']);self.step=state['step']
        self.samples.rng.setstate(state['sampler_rng']);random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])


def binding_for(config,path):
    code=[Path(__file__),Path(base.__file__)]+[Path(__file__).with_name(x) for x in (
      'coherent_surface_model.py','coherent_surface_prior.py','coherent_shared_model.py','engine.py','runtime.py')]
    return {'code':{str(p):file_hash(p) for p in code},'config_sha256':file_hash(path),
      'manifest_sha256':file_hash(config['manifest']),'prior_sha256':file_hash(config['prior']),
      'initial_weights_sha256':file_hash(config['initial_weights']),
      'hardware_contract_sha256':file_hash(r'E:\fpga\network_ab\model_b_contract.json'),
      'role':'experimental_surface_teacher_not_hardware_contract'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int,default=200)
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8'));root=configure_workspace()
    if (root/'runs').resolve() not in a.run.resolve().parents:raise ValueError('Wrong run root')
    if (a.run/'run.json').exists() and not a.resume:raise FileExistsError(a.run)
    binding=binding_for(config,a.config);a.run.mkdir(exist_ok=True,parents=True)
    train=SurfaceTraining(config);source=a.run/'source';source.mkdir(exist_ok=True)
    for path in [Path(k) for k in binding['code']]+[a.config]:shutil.copy2(path,source/path.name)
    write_json(a.run/'run.json',{'binding':binding,'config':config,'precision':'FP32-no-AMP-no-TF32',
      'parameters':sum(p.numel() for p in train.net.parameters())})
    if a.preflight:
        with torch.no_grad():
            probe=torch.rand(1,3,128,192,device='cuda')*1.9-.95
            initial_error=float((train.net(probe)-train.anchor(probe)).abs().max())
        if initial_error!=0:raise ValueError('Warm start is not exact')
        initial=a.run/'stage_initial.pt';torch.save(train.state(binding),initial)
        rows=[train.update() for _ in range(3)]
        saved=a.run/'resume.pt';torch.save(train.state(binding),saved)
        ref=train.update();expected={k:v.detach().cpu().clone() for k,v in train.net.state_dict().items()}
        train.restore(torch.load(saved,map_location='cpu',weights_only=True),binding);replay=train.update()
        error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in train.net.state_dict().items())
        with torch.no_grad():full=train.ema(torch.zeros(1,3,720,1280,device='cuda'))
        if error>1e-5 or full.shape!=(1,3,720,1280) or not torch.isfinite(full).all():raise RuntimeError('GPU preflight failed')
        write_json(a.run/'preflight.json',{'status':'PASS','binding':binding,'initial_max_abs_error':initial_error,
          'steps':rows,'reference':ref,'replay':replay,'resume_max_abs_error':error,
          'full_720_shape':list(full.shape),'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2})
        print(json.dumps({'status':'PASS','mode':config['mode'],'initial_error':initial_error,'resume_error':error}),flush=True)
        return
    check=json.loads((Path(config['preflight_run'])/'preflight.json').read_text())
    if check['status']!='PASS' or check['binding']!=binding:raise ValueError('GPU preflight mismatch')
    if a.resume:train.restore(torch.load(a.resume,map_location='cpu',weights_only=True),binding)
    log=a.run/f'metrics_from_{train.step:06d}.jsonl'
    try:
        while train.step<a.stop_after:
            row=train.update()
            if not all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):raise FloatingPointError(row)
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            if train.step%10==0 or train.step==1:
                write_json(a.run/'progress.json',{'status':'RUNNING',**row})
                print(json.dumps({k:row[k] for k in ('step','mode','target','detail','temporal','seconds')}),flush=True)
            if train.step in config['preview_steps']:base.previews(train,a.run)
            if train.step%config['checkpoint_every']==0 or train.step==a.stop_after:
                if binding_for(config,a.config)!=binding:raise ValueError('Run binding drift')
                path=a.run/f'checkpoint_{train.step:06d}.pt';tmp=path.with_suffix('.tmp')
                torch.save(train.state(binding),tmp);os.replace(tmp,path)
        write_json(a.run/'status.json',{'status':'STOP_AFTER_REACHED','step':train.step,
          'quality_gate':'UNREVIEWED','checkpoint':str(path),'sha256':file_hash(path)})
    except Exception as exc:
        write_json(a.run/'status.json',{'status':'FAILED','step':train.step,'error':str(exc)});raise


if __name__=='__main__':main()
