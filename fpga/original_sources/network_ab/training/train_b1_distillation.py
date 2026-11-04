"""Bounded FP32 corrected-teacher distillation into the unchanged B1 contract."""
import argparse
import copy
import gc
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from network_ab.contract_torch import build_fp32_model
from .optimization_policy import initialize_rgb_path
from .coherent_shared_model import warp
from .engine import set_reproducible
from .losses import PerceptualLoss
from .runtime import configure_workspace, contract_guard, file_hash, write_json


def as_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).float()/128-1


def to_rgb(x):
    return (x.detach().cpu().permute(1,2,0).numpy()*128+128).round().clip(0,255).astype(np.uint8)


@lru_cache(maxsize=96)
def read_image(path):
    return np.array(Image.open(path).convert('RGB'))


@lru_cache(maxsize=32)
def read_flow(path):
    with np.load(path) as z:
        flow=cv2.resize(z['backward'].astype(np.float32),(1280,720))*2
        mask=cv2.resize(z['mask'].astype(np.float32),(1280,720),interpolation=cv2.INTER_NEAREST)
    return flow,mask


def edges(x):
    return x[:,:,1:]-x[:,:,:-1], x[:,:,:,1:]-x[:,:,:,:-1]


class Training:
    def __init__(self, root, config, manifest):
        contract_guard(); self.step=0; self.config=config; self.manifest=manifest
        self.rng=random.Random(config['seed']); self.groups={}
        for row in manifest['pairs']:
            assert manifest['source_splits'][row['source']]['split']=='train'
            self.groups.setdefault(row['source'],[]).append(row)
        torch.manual_seed(config['seed'])
        self.net=build_fp32_model().cuda()
        initialize_rgb_path(self.net)
        self.net.eval() # Freeze running statistics; affine remains trainable.
        self.ema=copy.deepcopy(self.net).requires_grad_(False)
        self.opt=torch.optim.Adam(self.net.parameters(),lr=config['lr'])
        self.vgg=PerceptualLoss(root).cuda(); self.vgg.features=self.vgg.features[:9]

    def batch(self, native=False):
        values=[];flows=[];masks=[];spec=[]
        for _ in range(1 if native else self.config['pairs_per_step']):
            row=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))])
            if native: x,y,w,h=0,0,1280,720
            else:
                w=h=self.config['crop'];x=4*self.rng.randrange((1280-w)//4+1);y=4*self.rng.randrange((720-h)//4+1)
            values.extend([as_tensor(read_image(row[k])[y:y+h,x:x+w]) for k in ['current','next','current_target','next_target']])
            flow,mask=read_flow(row['flow']);f=flow[y:y+h,x:x+w].copy();m=mask[y:y+h,x:x+w].copy()
            yy,xx=np.mgrid[:h,:w];margin=self.config['margin']
            m*=((xx+f[:,:,0]>=margin)&(xx+f[:,:,0]<w-margin)&(yy+f[:,:,1]>=margin)&(yy+f[:,:,1]<h-margin))
            flows.append(torch.from_numpy(f).permute(2,0,1));masks.append(torch.from_numpy(m)[None])
            spec.append({'source':row['source'],'index':row['current_index'],'xy':[x,y],'wh':[w,h]})
        n=len(spec);v=torch.stack(values).cuda().reshape(n,4,3,h,w)
        return v[:,0],v[:,1],v[:,2],v[:,3],torch.stack(flows).cuda(),torch.stack(masks).cuda(),spec

    def update(self, native=None):
        begin=time.perf_counter();self.net.eval()
        if native is None:native=(self.step+1)%self.config['native_interval']==0
        a,b,ta,tb,flow,mask,spec=self.batch(native)
        count=len(a);src=torch.cat([a,b]);target=torch.cat([ta,tb]);pred=self.net(src)
        m=self.config['margin'];cut=lambda v:v[:,:,m:-m,m:-m]
        p,t=cut(pred),cut(target)
        hp=lambda v:v-F.avg_pool2d(v,5,1,2)
        # Perceptual comparison is against the style target, not source identity.
        pv,tv=(F.avg_pool2d(p,2),F.avg_pool2d(t,2)) if native else (p,t)
        with torch.no_grad():tf=self.vgg.extract(tv)['relu2_2']
        pf=self.vgg.extract(pv)['relu2_2']
        delta=(pred[count:]-b)-warp(pred[:count]-a,flow)
        mc=cut(mask)
        terms={'paired':F.l1_loss(p,t),'edge':sum(F.l1_loss(x,y) for x,y in zip(edges(p),edges(t))),
               'detail':F.l1_loss(hp(p),hp(t)), 'perceptual':F.l1_loss(pf,tf),
               'temporal':(cut(delta).abs()*mc).sum()/(mc.sum()*3+1e-6)}
        loss=sum(terms[k]*v for k,v in self.config['weights'].items())
        if not torch.isfinite(loss):raise FloatingPointError('Loss')
        progress=self.step/max(1,self.config['max_steps']-1)
        lr=(self.config['lr_floor']+(self.config['lr']-self.config['lr_floor'])*.5*(1+math.cos(math.pi*progress)))*min(1,(self.step+1)/25)
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for name,param in self.net.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():raise FloatingPointError(name)
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True)
        if norm<=0:raise FloatingPointError('No gradient')
        self.opt.step();self.step+=1
        with torch.no_grad():
            for dst,src in zip(self.ema.parameters(),self.net.parameters()):dst.lerp_(src,.05)
            for dst,src in zip(self.ema.buffers(),self.net.buffers()):dst.copy_(src)
        torch.cuda.synchronize()
        return {'step':self.step,'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},
                'lr':lr,'grad_norm':float(norm),'seconds':time.perf_counter()-begin,
                'peak_MiB':torch.cuda.max_memory_allocated()/2**20,'native':native,'samples':spec}

    def state(self,binding):
        return {'schema':'b1-fp32-distillation-checkpoint-v1','step':self.step,'config':self.config,'binding':binding,
                'model':self.net.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.opt.state_dict(),
                'sampler_rng':self.rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}

    def restore(self,state,binding):
        assert state['schema']=='b1-fp32-distillation-checkpoint-v1' and state['config']==self.config and state['binding']==binding
        self.net.load_state_dict(state['model']);self.ema.load_state_dict(state['ema']);self.opt.load_state_dict(state['optimizer'])
        self.rng.setstate(state['sampler_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);self.step=state['step']


def validate(net,manifest,out):
    out.mkdir(parents=True,exist_ok=True);records=[];panels=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            x=as_tensor(read_image(row['source']))[None].cuda();t=as_tensor(read_image(row['target']))[None].cuda();y=net(x)
            rgb=to_rgb(y[0]);Image.fromarray(rgb).save(out/(row['id']+'.png'))
            records.append({'id':row['id'],'teacher_mae_rgb255':float((y-t).abs().mean()*128),
                            'identity_teacher_mae_rgb255':float((x-t).abs().mean()*128),
                            'source_change_rgb255':float((y-x).abs().mean()*128),
                            'edge_target_mae_rgb255':sum(float((u-v).abs().mean()*128) for u,v in zip(edges(y),edges(t)))/2})
            panel=Image.new('RGB',(960,204),'#111');draw=ImageDraw.Draw(panel)
            for i,(a,label) in enumerate([(read_image(row['source']),'SOURCE'),(read_image(row['target']),'TEACHER + CORRECTION'),(rgb,'B1 FP32')]):
                panel.paste(Image.fromarray(a).resize((320,180)),(320*i,24));draw.text((320*i+5,5),label,fill='white')
            panels.append(panel)
    board=Image.new('RGB',(960,204*len(panels)))
    for i,p in enumerate(panels):board.paste(p,(0,i*204))
    board.save(out/'comparison.jpg',quality=94)
    result={'records':records,'mean_teacher_mae_rgb255':sum(r['teacher_mae_rgb255'] for r in records)/len(records),
            'note':'Source-isolated previously reviewed regressions, no blind-quality claim'}
    write_json(out/'report.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','train']);ap.add_argument('--resume',type=Path);ap.add_argument('--stop-after',type=int,default=2000);args=ap.parse_args()
    root=configure_workspace();run=root/'runs/b1-distillation-20261003';config=json.loads((run/'config.json').read_text())
    set_reproducible(config['seed']);cv2.setNumThreads(2)
    manifest=json.loads((run/'data/manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='REVIEWED_FOR_B1_FP32_DISTILLATION'
    for row in manifest['pairs']:
        for key in ['current','next','current_target','next_target','flow']:assert file_hash(row[key])==row[key+'_sha256']
    code=[Path(__file__),Path(__file__).with_name('fpga_video_resize.py'),Path(__file__).with_name('optimization_policy.py'),Path(__file__).with_name('losses.py'),Path(__file__).with_name('coherent_shared_model.py'),Path(__file__).parents[1]/'contract_torch.py',Path(__file__).parents[1]/'model_b_contract.json']
    binding={'manifest_sha256':file_hash(run/'data/manifest.json'),'config_sha256':file_hash(run/'config.json'),
             'code':{str(p):file_hash(p) for p in code},'dtype':'FP32','tf32':False}
    t=Training(root,config,manifest)
    if args.mode=='preflight':
        out=run/'preflight';out.mkdir(exist_ok=False)
        rows=[t.update(native=True),t.update(native=False)];torch.save(t.state(binding),out/'resume.pt')
        ref=t.update(native=False);expected={k:v.cpu().clone() for k,v in t.net.state_dict().items()}
        del t;gc.collect();torch.cuda.empty_cache()
        t=Training(root,config,manifest);t.restore(torch.load(out/'resume.pt',map_location='cpu',weights_only=True),binding)
        replay=t.update(native=False);err=max(float((v.cpu()-expected[k]).abs().max()) for k,v in t.net.state_dict().items())
        assert ref['samples']==replay['samples'] and err<2e-6
        assert all(p.dtype==torch.float32 for p in t.net.parameters())
        write_json(out/'report.json',{'status':'PASS','steps':rows,'replay_error':err,'reference':ref,'replay':replay,'binding':binding})
        print('PREFLIGHT_PASS',err,flush=True);return
    pre=json.loads((run/'preflight/report.json').read_text());assert pre['status']=='PASS' and pre['binding']==binding
    out=run/'train';out.mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:raise FileExistsError('Resume explicitly')
    if args.resume:t.restore(torch.load(args.resume,map_location='cpu',weights_only=True),binding)
    else:validate(t.ema,manifest,out/'validation_000000')
    frozen=out/'source';frozen.mkdir(exist_ok=True)
    for p in code:shutil.copy2(p,frozen/p.name)
    write_json(out/'run.json',{'status':'RUNNING','config':config,'binding':binding,'resumed_from':str(args.resume)})
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as stream:
        while t.step<min(args.stop_after,config['max_steps']):
            row=t.update();stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if t.step%25==0:write_json(out/'progress.json',row);print(json.dumps({k:row[k] for k in ['step','loss','paired','edge','temporal','seconds','peak_MiB']}),flush=True)
            if t.step%250==0:
                torch.save(t.state(binding),out/f'checkpoint_{t.step:06d}.pt')
                scores=validate(t.ema,manifest,out/f'validation_{t.step:06d}')
                print('VALIDATE',t.step,scores['mean_teacher_mae_rgb255'],flush=True)
        if t.step%250:torch.save(t.state(binding),out/f'checkpoint_{t.step:06d}.pt')
    write_json(out/'status.json',{'status':'COMPLETED_BOUNDED_FP32_DISTILLATION','step':t.step,'requires_video_review':True})


if __name__=='__main__':main()
