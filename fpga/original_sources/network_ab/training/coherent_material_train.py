"""Train material features from real game targets, anchored to stable 4850 CNN."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time
import torch
from torch.nn import functional as F
from . import coherent_shared_train as base
from .coherent_surface_train import SurfaceTraining,smooth
from .coherent_surface_model import SurfaceAppearanceNet
from .coherent_shared_model import warp
from .coherent_material_loss import texture_view
from .coherent_material_match import reference_targets,fixed_material_loss
from .coherent_surface_prior import luminance
from .losses import PerceptualLoss
from .engine import set_reproducible
from .runtime import configure_workspace,file_hash,write_json


class MaterialTraining(SurfaceTraining):
    def __init__(self,config):
        root=configure_workspace();set_reproducible(config['seed']);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.config=config;self.samples=base.Samples(config['manifest'],config['seed'])
        state=torch.load(config['initial_weights'],map_location='cpu',weights_only=True)
        if state['schema']!='surface-appearance-fp32-weights-v1' or state['training_steps']!=config['prior_training_steps']:
            raise ValueError('Wrong initial surface teacher')
        if state['config']!=config['model'] or state['binding']['manifest_sha256']!=file_hash(config['manifest']):
            raise ValueError('Model or data drift')
        self.net=SurfaceAppearanceNet(**config['model']).cuda();self.net.load_state_dict(state['generator'])
        self.anchor=copy.deepcopy(self.net).eval().requires_grad_(False)
        self.ema=copy.deepcopy(self.anchor)
        self.vgg=PerceptualLoss(root).cuda()
        bank=torch.load(config['material_bank'],map_location='cpu',weights_only=True)
        if bank['binding']['manifest_sha256']!=file_hash(config['manifest']) or bank['binding']['vgg_sha256']!=self.vgg.weights_sha256:
            raise ValueError('Material bank data mismatch')
        if bank['binding']['loss_code_sha256']!=file_hash(Path(__file__).with_name('coherent_material_loss.py')):
            raise ValueError('Material representation drift')
        self.bank={k:v.cuda() for k,v in bank['bank'].items()}
        self.opt=torch.optim.Adam(self.net.parameters(),lr=config['learning_rate'],betas=(.9,.999));self.step=0

    def update(self):
        started=time.perf_counter();self.net.train()
        a,b,flow,mask,_,spec=self.samples.next();inputs=torch.cat((a,b))
        with torch.no_grad():
            anchored=self.anchor(inputs);aa,ab=anchored.chunk(2)
            reference=self.vgg.extract(texture_view(aa))
            targets=reference_targets(reference,self.bank)
        output=self.net(inputs);oa,ob=output.chunk(2)
        features=self.vgg.extract(texture_view(oa))
        delta=output-anchored;da,db=delta.chunk(2)
        temporal=db-warp(da,flow)
        yl,al=luminance(output),luminance(anchored)
        edge=[]
        for d,e in zip(base.differences(smooth(yl,3)),base.differences(smooth(al,3))):
            keep=(e.abs()>.06).detach()
            edge.append(((d-e).abs()*keep).sum()/(keep.sum()+1))
        terms={'material':fixed_material_loss(features,targets),
          'content':base.relative_mse(features['relu4_3'],reference['relu4_3']),
          'tone':F.l1_loss(F.avg_pool2d(output,16),F.avg_pool2d(anchored,16)),
          'chroma':F.l1_loss(output-yl,anchored-al),'anchor':delta.abs().mean(),
          'edge':sum(edge),'temporal':(temporal.abs()*mask).sum()/(mask.sum()*3+1e-6)}
        weights=self.config['loss_weights'];loss=sum(weights[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite material loss')
        self.opt.zero_grad(set_to_none=True);loss.backward();shift_value=0.
        if self.step%4==0:
            shifted=torch.roll(a,1,3)
            with torch.no_grad():
                target_delta=torch.roll(self.net(a)-self.anchor(a),1,3)
                shifted_anchor=self.anchor(shifted)
            shift=F.l1_loss((self.net(shifted)-shifted_anchor)[:,:,12:-12,12:-12],target_delta[:,:,12:-12,12:-12])
            (weights['shift']*shift).backward();shift_value=float(shift.detach())
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),10,error_if_nonfinite=True)
        if float(grad)<=0:raise FloatingPointError('Zero gradient')
        self.opt.step()
        with torch.no_grad():
            decay=min(.98,(1+self.step)/(10+self.step))
            for dst,src in zip(self.ema.parameters(),self.net.parameters()):dst.lerp_(src,1-decay)
        self.step+=1
        return {'step':self.step,**{k:float(v.detach()) for k,v in terms.items()},'shift':shift_value,
          'gradient':float(grad),'seconds':time.perf_counter()-started,
          'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2,'sample':spec,
          'style_supervision':'41-image frozen feature bank; anchor-fixed correspondence, no output rematching'}


def binding_for(config,path):
    names=('coherent_surface_train.py','coherent_shared_train.py','coherent_surface_model.py',
      'coherent_shared_model.py','coherent_material_loss.py','coherent_material_match.py',
      'coherent_surface_prior.py','losses.py','engine.py','runtime.py')
    code=[Path(__file__)]+[Path(__file__).with_name(n) for n in names]
    return {'code':{str(p):file_hash(p) for p in code},'config_sha256':file_hash(path),
      'manifest_sha256':file_hash(config['manifest']),'material_bank_sha256':file_hash(config['material_bank']),
      'initial_weights_sha256':file_hash(config['initial_weights']),
      'hardware_contract_sha256':file_hash(r'E:\fpga\network_ab\model_b_contract.json'),
      'role':'experimental_material_teacher_not_hardware_contract'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int,default=300)
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8'));root=configure_workspace()
    if (root/'runs').resolve() not in a.run.resolve().parents:raise ValueError('Wrong run root')
    if (a.run/'run.json').exists() and not a.resume:raise FileExistsError(a.run)
    binding=binding_for(config,a.config);a.run.mkdir(parents=True,exist_ok=True)
    train=MaterialTraining(config);source=a.run/'source';source.mkdir(exist_ok=True)
    for path in [Path(k) for k in binding['code']]+[a.config]:shutil.copy2(path,source/path.name)
    write_json(a.run/'run.json',{'binding':binding,'config':config,'precision':'FP32-no-AMP-no-TF32','parameters':sum(p.numel() for p in train.net.parameters())})
    if a.preflight:
        with torch.no_grad():
            probe=torch.rand(1,3,128,192,device='cuda')*1.9-.95;initial=float((train.net(probe)-train.anchor(probe)).abs().max())
        if initial!=0:raise ValueError('Initial output mismatch')
        torch.save(train.state(binding),a.run/'stage_initial.pt')
        rows=[train.update() for _ in range(3)];saved=a.run/'resume.pt';torch.save(train.state(binding),saved)
        ref=train.update();expected={k:v.detach().cpu().clone() for k,v in train.net.state_dict().items()}
        train.restore(torch.load(saved,map_location='cpu',weights_only=True),binding);replay=train.update()
        error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in train.net.state_dict().items())
        with torch.no_grad():full=train.ema(torch.zeros(1,3,720,1280,device='cuda'))
        if error>1e-5 or full.shape!=(1,3,720,1280) or not torch.isfinite(full).all():raise RuntimeError('Preflight failed')
        write_json(a.run/'preflight.json',{'status':'PASS','binding':binding,'initial_error':initial,'steps':rows,'reference':ref,'replay':replay,'resume_max_abs_error':error,'full_720_shape':list(full.shape)})
        print(json.dumps({'status':'PASS','resume_error':error,'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2}),flush=True);return
    check=json.loads((Path(config['preflight_run'])/'preflight.json').read_text())
    if check['status']!='PASS' or check['binding']!=binding:raise ValueError('Preflight mismatch')
    if a.resume:train.restore(torch.load(a.resume,map_location='cpu',weights_only=True),binding)
    log=a.run/f'metrics_from_{train.step:06d}.jsonl'
    try:
        while train.step<a.stop_after:
            row=train.update()
            if not all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):raise FloatingPointError(row)
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            if train.step%10==0 or train.step==1:
                write_json(a.run/'progress.json',{'status':'RUNNING',**row});print(json.dumps({k:row[k] for k in ('step','material','tone','temporal','seconds')}),flush=True)
            if train.step in config['preview_steps']:base.previews(train,a.run)
            if train.step%config['checkpoint_every']==0 or train.step==a.stop_after:
                if binding_for(config,a.config)!=binding:raise ValueError('Run drift')
                path=a.run/f'checkpoint_{train.step:06d}.pt';tmp=path.with_suffix('.tmp');torch.save(train.state(binding),tmp);os.replace(tmp,path)
        write_json(a.run/'status.json',{'status':'STOP_AFTER_REACHED','step':train.step,'checkpoint':str(path),'sha256':file_hash(path)})
    except Exception as e:write_json(a.run/'status.json',{'status':'FAILED','step':train.step,'error':str(e)});raise


if __name__=='__main__':main()
