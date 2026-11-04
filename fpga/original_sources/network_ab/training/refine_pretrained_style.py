"""Native-720 FP32 refinement of a pinned pretrained style network.

Encoder/attention stay frozen. Decoder learns reviewed smooth-surface corrections
and motion consistency. This is an experimental animation teacher, not FPGA B1.
"""
import argparse
import copy
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
from .trainable_animegan import TrainableAnimeGAN
from .animegan_teacher import AnimeTeacher, MODEL_SHA
from .train_clear_style import tensor, differences, Samples
from .coherent_shared_model import warp
from .engine import set_reproducible
from .runtime import configure_workspace, file_hash, write_json


def pixels(y):
    return (y[0].permute(1,2,0).detach().cpu().numpy()*128+128).clip(0,255).astype(np.uint8)


class Refine:
    def __init__(self,root,config):
        self.config=config;self.step=0
        self.samples=Samples(config['manifest'],config['seed'])
        self.net=TrainableAnimeGAN(root).cuda().eval()
        for name,p in self.net.named_parameters():
            p.requires_grad_('/main/Conv' in name)
        self.trainable=[p for p in self.net.parameters() if p.requires_grad]
        self.opt=torch.optim.Adam(self.trainable,lr=config['learning_rate'])
        self.ema=copy.deepcopy(self.net).requires_grad_(False)

    def update(self):
        start=time.perf_counter();rng=self.samples.rng
        row=rng.choice(self.samples.groups[rng.choice(sorted(self.samples.groups))])
        a,b,ta,tb=[tensor(self.samples.image(row[k])) for k in ['current','next','current_target','next_target']]
        with np.load(row['flow']) as z:
            fl=cv2.resize(z['backward'].astype(np.float32),(1280,720))*2
            ma=cv2.resize(z['mask'].astype(np.float32),(1280,720),interpolation=cv2.INTER_NEAREST)
        yy,xx=np.mgrid[:720,:1280]
        ma*=((xx+fl[:,:,0]>=8)&(xx+fl[:,:,0]<1272)&(yy+fl[:,:,1]>=8)&(yy+fl[:,:,1]<712))
        flow=torch.from_numpy(fl.copy()).permute(2,0,1)[None].cuda();mask=torch.from_numpy(ma.copy())[None,None].cuda()
        self.opt.zero_grad(set_to_none=True)
        # Two full images per optimizer step, processed sequentially to bound VRAM.
        # Both frames get paired supervision; the next frame gets the causal temporal
        # gradient against a detached previous output. No image is resized/cropped.
        with torch.no_grad():previous=self.net(a)
        out=self.net(b);hp=lambda v:v-F.avg_pool2d(v,5,1,2)
        temporal=((out-b)-warp(previous-a,flow)).abs()
        terms={'paired':F.l1_loss(out,tb),'detail':F.l1_loss(hp(out),hp(tb)),
               'edge':sum(F.l1_loss(u,v) for u,v in zip(differences(out),differences(tb))),
               'temporal':(temporal*mask).sum()/(mask.sum()*3+1e-6)}
        loss=sum(terms[k]*v for k,v in self.config['weights'].items())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();values={k:float(v.detach()) for k,v in terms.items()}
        # First-frame paired supervision is accumulated separately (effective batch two).
        del out,previous,temporal,loss,terms
        out=self.net(a)
        first=F.l1_loss(out,ta)*self.config['weights']['paired']+F.l1_loss(hp(out),hp(ta))*self.config['weights']['detail']
        first+=sum(F.l1_loss(u,v) for u,v in zip(differences(out),differences(ta)))*self.config['weights']['edge']
        first.backward()
        for p in self.trainable:p.grad.mul_(.5)
        grad=torch.nn.utils.clip_grad_norm_(self.trainable,5,error_if_nonfinite=True)
        if not grad>0:raise FloatingPointError('Zero gradient')
        self.opt.step();self.step+=1
        with torch.no_grad():
            for dst,src in zip(self.ema.parameters(),self.net.parameters()):dst.lerp_(src,.1)
        torch.cuda.synchronize()
        return {'step':self.step,**values,'grad_norm':float(grad),'seconds':time.perf_counter()-start,
                'peak_MiB':torch.cuda.max_memory_allocated()/2**20,
                'sample':{'source':row['source'],'parent_pair_index':row['parent_pair_index']}}

    def state(self,binding):
        return {'schema':'pretrained-style-refinement-v1','step':self.step,'config':self.config,'binding':binding,
                'generator':self.net.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.opt.state_dict(),
                'sampler_rng':self.samples.rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}

    def restore(self,state,binding):
        if state['config']!=self.config or state['binding']!=binding:raise ValueError('Resume binding drift')
        self.net.load_state_dict(state['generator']);self.ema.load_state_dict(state['ema']);self.opt.load_state_dict(state['optimizer'])
        self.step=state['step'];self.samples.rng.setstate(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])


def validate(net,root,out):
    out.mkdir(exist_ok=True);records=[];parent=root/'runs/clear-style-control-20261003'
    with torch.no_grad():
        for case in ['quiet_stairs','plaza_stone','park_people']:
            a=np.array(Image.open(parent/'anime'/case/'source.png'));y=net(tensor(a));b=pixels(y)
            target=np.array(Image.open(parent/'guard_review'/(case+'.png')))
            Image.fromarray(b).save(out/(case+'.png'))
            records.append({'case':case,'source_mae':float(np.abs(b.astype(float)-a).mean()),
                            'guard_target_mae':float(np.abs(b.astype(float)-target).mean())})
    write_json(out/'report.json',{'records':records,'note':'Reviewed held-out scenes; not blind evaluation or aesthetic acceptance'})


def conversion(root,out):
    out.mkdir(exist_ok=True);net=TrainableAnimeGAN(root).cuda().eval();records=[]
    parent=root/'runs/clear-style-control-20261003'
    with torch.no_grad():
        for case in ['quiet_stairs','plaza_stone','park_people']:
            x=np.array(Image.open(parent/'anime'/case/'source.png'));y=pixels(net(tensor(x)))
            ref=np.array(Image.open(parent/'anime'/case/'hayao_author.png'));diff=np.abs(y.astype(float)-ref)
            assert diff.max()<=1 and diff.mean()<.002
            Image.fromarray(y).save(out/(case+'.png'))
            records.append({'case':case,'shape':list(y.shape),'mae':float(diff.mean()),'max':float(diff.max())})
    teacher=AnimeTeacher(root)
    x=np.array(Image.open(parent/'anime/quiet_stairs/source.png'))[:256,:384].copy()
    with torch.no_grad():y=net(tensor(x));actual=(y[0].permute(1,2,0).cpu().numpy()+1)*128
    inp=np.ascontiguousarray(x[None],dtype=np.float32)/127.5-1
    ref=(teacher.session.run(None,{teacher.input:inp})[0][0]+1)*127.5
    diff=np.abs(actual-ref);assert diff.max()<.03 and diff.mean()<.003
    records.append({'case':'256x384_float','shape':list(actual.shape),'mae':float(diff.mean()),'max':float(diff.max())})
    write_json(out/'verification.json',{'status':'PASS','records':records,'author_sha256':MODEL_SHA,
              'parameters':sum(p.numel() for p in net.parameters()),'float32':True,'tf32':False,
              'normalization_affine_constants':'ONNX initializer buffers remain fixed; convolutions are trainable'})
    print('CONVERSION_PASS',records,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['conversion','preflight','train']);ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=240);args=ap.parse_args()
    root=configure_workspace();run=root/'runs/texture-recovery-20261003';set_reproducible(202610032);cv2.setNumThreads(2)
    if args.mode=='conversion':conversion(root,run/'conversion');return
    config=json.loads((run/'refine_config.json').read_text(encoding='utf-8'))
    code=[Path(__file__),Path(__file__).with_name('trainable_animegan.py'),Path(__file__).with_name('animegan_teacher.py'),Path(__file__).with_name('train_clear_style.py'),Path(__file__).with_name('coherent_shared_model.py')]
    binding={'code':{str(p):file_hash(p) for p in code},'manifest_sha256':file_hash(config['manifest']),
             'config_sha256':file_hash(run/'refine_config.json'),'author_sha256':MODEL_SHA,'dtype':'FP32','tf32':False}
    assert json.loads((run/'conversion/verification.json').read_text())['status']=='PASS'
    t=Refine(root,config)
    if args.mode=='preflight':
        out=run/'refine_preflight';out.mkdir(exist_ok=True)
        rows=[t.update() for _ in range(2)];saved=out/'resume.pt';torch.save(t.state(binding),saved)
        ref=t.update();expected={k:v.detach().cpu().clone() for k,v in t.net.state_dict().items()}
        del t;gc.collect();torch.cuda.empty_cache()
        t=Refine(root,config);t.restore(torch.load(saved,map_location='cpu',weights_only=True),binding);replay=t.update()
        errs=[(v.detach().cpu()-expected[k]).abs().max().item() for k,v in t.net.state_dict().items() if v.numel()]
        err=max(errs);assert ref['sample']==replay['sample'] and err<=2e-6
        assert all(p.dtype==torch.float32 for p in t.net.parameters())
        write_json(out/'report.json',{'status':'PASS','steps':rows,'reference':ref,'replay':replay,'resume_max_error':err,
                   'trainable_parameters':sum(p.numel() for p in t.trainable),'binding':binding})
        print('PREFLIGHT_PASS',err,'seconds',replay['seconds'],'peak',replay['peak_MiB'],flush=True);return
    pre=json.loads((run/'refine_preflight/report.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    out=run/'refine_train';out.mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:raise FileExistsError('Explicit resume required')
    if args.resume:t.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
    frozen=out/'source';frozen.mkdir(exist_ok=True)
    for p in code:shutil.copy2(p,frozen/p.name)
    shutil.copy2(run/'refine_config.json',frozen/'refine_config.json')
    write_json(out/'run.json',{'status':'RUNNING','config':config,'binding':binding,'resumed_from':str(args.resume)})
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while t.step<min(args.stop_after,config['max_steps']):
            row=t.update();f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if t.step%10==0:write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
            if t.step%40==0:
                torch.save(t.state(binding),out/f'checkpoint_{t.step:06d}.pt');validate(t.ema,root,out/f'validation_{t.step:06d}')
    write_json(out/'status.json',{'status':'COMPLETED_BOUNDED_REFINEMENT','step':t.step,'requires_video_review':True})


if __name__=='__main__':main()
