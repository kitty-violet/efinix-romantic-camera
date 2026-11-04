"""Controlled recovery experiments; inference is always the unchanged B1 graph.

Frequency weighting follows the ICCV2021 FFL mathematical formulation, written
here with complex FFT tensors. Auxiliary polyphase heads exist only in training.
No external restoration, test-image fitting, new deployment ops or downloads.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import train_b1_reconstruction as r
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible

OUT = r.ROOT/'model-training-workspace/runs/b1-reconstruction-research-20261005'


def ssim_map(a, b):
    # RGB normalized to [0,1]; local uniform-window SSIM, not MS-SSIM.
    a, b = (a+1)/2, (b+1)/2
    ma, mb = F.avg_pool2d(a, 7, 1), F.avg_pool2d(b, 7, 1)
    va = (F.avg_pool2d(a*a, 7, 1)-ma*ma).clamp_min(0)
    vb = (F.avg_pool2d(b*b, 7, 1)-mb*mb).clamp_min(0)
    cov = F.avg_pool2d(a*b, 7, 1)-ma*mb
    return ((2*ma*mb+.01**2)*(2*cov+.03**2)/
            ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))


def focal_frequency(a, b):
    # Local 2x2 FFT patches; retain phase as well as amplitude.
    h, w = a.shape[-2:]
    a, b = a[..., :h//2*2, :w//2*2], b[..., :h//2*2, :w//2*2]
    def split(z):
        n,c,h,w = z.shape
        return z.reshape(n,c,2,h//2,2,w//2).permute(0,2,4,1,3,5)
    delta = torch.fft.fft2(split(a), norm='ortho')-torch.fft.fft2(split(b), norm='ortho')
    distance = delta.abs()
    weights = (distance/(distance.amax((-1,-2),keepdim=True)+1e-8)).detach()
    return (weights*distance.square()).mean()


def gradients(a, b):
    return sum(F.l1_loss(torch.diff(a,dim=d),torch.diff(b,dim=d)) for d in (-1,-2))/2


def loss_values(p, t, mode):
    vals = dict(mse=F.mse_loss(p,t), l1=F.l1_loss(p,t))
    if mode == 'mse':
        return vals['mse'], vals
    vals['edge'] = gradients(p,t)
    if mode in ('frequency','combined'):
        vals['frequency'] = focal_frequency(p,t)
    if mode in ('structure','combined'):
        vals['ssim'] = 1-ssim_map(p,t).mean()
    loss = vals['mse']+.25*vals['l1']+.15*vals['edge']
    if 'frequency' in vals:
        loss = loss+vals['frequency']
    if 'ssim' in vals:
        loss = loss+.10*vals['ssim']
    return loss, vals


@torch.no_grad()
def independent_init(net, seed):
    torch.manual_seed(seed)
    for key,w in net.weight.items():
        if key.startswith('res'):
            if key.endswith('dw'):
                w.zero_();w[:,0,1,1]=1
            elif 'conv1' in key:
                nn.init.orthogonal_(w.flatten(1),gain=.5)
            else:
                w.normal_(0,.0005)
            net.bias[key].zero_()
        elif w.shape[-1] == 3 and w.shape[1] == 1:
            w.normal_(0,.035)
            for i in range(len(w)):
                dy,dx = ((i//2)%2,i%2) if key.startswith('down') else (0,0)
                w[i,0,1+dy,1+dx] += .8
            net.bias[key].fill_(5)
        else:
            nn.init.orthogonal_(w.flatten(1),gain=.7)
            net.bias[key].fill_(8 if key != 'rgb_head' else 0)
    # Independent positive/negative colour channels, plus luminance derivatives.
    w=net.weight['stem'];w.zero_();net.bias['stem'].zero_()
    for i in range(6):
        w[i,i//2,1,1] = .7 if i%2==0 else -.7
    sobel = torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]],device=w.device)/16
    for j in range(3):
        w[6,j]=sobel/3;w[7,j]=sobel.T/3


class LearnableQ17(r.c.ColourB1):
    """Training-only scale parameters; forward/export use integer Q17 as before."""
    def __getattr__(self,name):
        value=super().__getattr__(name)
        if name.endswith('_m') and name in self._parameters:
            return self.round(value).clamp(1,131071)
        return value


class Experiment:
    def __init__(self, args, rows):
        self.args=args;self.rows=rows;self.step=0
        self.rng=random.Random(args.seed)
        self.net=r.initialize(args.seed)
        warm_is_quantized=False
        if args.warm:
            parent=torch.load(args.warm,map_location='cpu',weights_only=True)
            self.net.load_state_dict(parent['model'])
            # Residual Q17 alignment belongs to the executable model, but is
            # package metadata rather than a state_dict tensor. Loading only
            # tensors silently retains the initializer's different alignment.
            if 'residual_encoding' in parent:
                self.net.package['residual_encoding']=copy.deepcopy(parent['residual_encoding'])
            warm_is_quantized=bool(parent.get('qat',False) or parent.get('args',{}).get('qat',False))
        elif args.init == 'independent':
            independent_init(self.net,args.seed)
        self.heads=nn.ModuleDict()
        if args.aux:
            self.heads['s2']=nn.Conv2d(8,12,1).cuda()
            self.heads['s4']=nn.Conv2d(16,48,1).cuda()
        self.perception=None
        if args.perceptual:
            from .losses import PerceptualLoss
            self.perception=PerceptualLoss(r.ROOT/'model-training-workspace').cuda()
            self.perception.features=self.perception.features[:16]
        self.net.quantized=args.qat
        # Continuing a trained integer model must preserve its saved Q17 scales.
        # Recalibrating from changed floating weights would change its function.
        if args.qat and not warm_is_quantized:
            self.net.calibrate()
        scale_parameters=[]
        if args.scale_lr:
            assert args.qat,'Learned Q17 scales require integer forward'
            self.net.__class__=LearnableQ17
            for key in self.net.weight:
                name=key+'_m';value=getattr(self.net,name).detach().clone()
                del self.net._buffers[name]
                self.net.register_parameter(name,nn.Parameter(value))
                scale_parameters.append(self.net._parameters[name])
        groups=[
            dict(params=list(self.net.weight.values()),lr=args.lr),
            dict(params=list(self.net.bias.values()),lr=args.lr*30),
            dict(params=list(self.heads.parameters()),lr=.001)]
        if args.scale_lr:groups.append(dict(params=scale_parameters,lr=args.scale_lr))
        self.opt=torch.optim.Adam(groups)
        self.base_lrs=[g['lr'] for g in self.opt.param_groups]
        self.groups={}
        for row in rows:self.groups.setdefault(row['video'],[]).append(row)

    def batch(self, native=False):
        images=[];specs=[]
        for _ in range(1 if native else self.args.batch):
            row=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))])
            a=r.read(row['path'])
            if native:
                x=y=0;h,w=720,1280;flip=0
            else:
                h=w=self.args.crop
                # Half uniform; half chooses the most textured of four crops.
                choices=[]
                for _ in range(4 if self.args.hard and self.rng.randrange(2) else 1):
                    y=self.rng.randrange(721-h);x=self.rng.randrange(1281-w)
                    patch=a[y:y+h:4,x:x+w:4].astype(np.float32)
                    score=float(np.abs(np.diff(patch,axis=0)).mean()+np.abs(np.diff(patch,axis=1)).mean())
                    choices.append((score,x,y))
                _,x,y=max(choices);flip=self.rng.randrange(2)
            v=a[y:y+h,x:x+w]
            if flip:v=v[:,::-1]
            rotations=self.rng.randrange(4) if self.args.rotate and not native else 0
            if rotations:v=np.rot90(v,rotations)
            permutation=[0,1,2]
            if self.args.colour and not native and self.rng.randrange(2):
                self.rng.shuffle(permutation)
                v=v[...,permutation]
            images.append(r.tensor(v));specs.append(dict(id=row['id'],xy=[x,y],wh=[w,h],flip=flip,rotations=rotations,colour_permutation=permutation))
        return torch.stack(images),specs

    def update(self,native=None):
        start=time.perf_counter()
        native=(self.step+1)%150==0 if native is None else native
        x,specs=self.batch(native)
        nodes=self.net(x,all_nodes=True)
        pred=nodes['rgb_head']/128
        # RF can extend ~40 pixels; don't train on synthetic crop boundaries.
        p,t=(pred,x) if native else (pred[...,40:-40,40:-40],x[...,40:-40,40:-40])
        loss,values=loss_values(p,t,self.args.loss)
        if self.args.shift:
            # Identity reconstruction should commute with translations. Exclude
            # wrapped borders and the receptive field; no temporal inference op.
            dy,dx=self.rng.choice(((0,1),(1,0),(1,1),(2,2)))
            shifted=self.net(torch.roll(x,(dy,dx),(-2,-1)))[0]
            aligned=torch.roll(shifted,(-dy,-dx),(-2,-1))
            margin=80 if native else 40
            equivariance=F.l1_loss(pred[...,margin:-margin,margin:-margin],aligned[...,margin:-margin,margin:-margin])
            loss=loss+self.args.shift*equivariance;values['shift_consistency']=equivariance
        if self.args.latent_refine:
            from .probe_b1_reconstruction_capacity import decoder
            latent=nodes['res3_add'].detach().clone().requires_grad_(True)
            latent_opt=torch.optim.Adam([latent],lr=.2)
            for _ in range(self.args.latent_refine):
                fitted=decoder(self.net,latent,detach_weights=True)
                fitted,target=(fitted,x) if native else (fitted[...,40:-40,40:-40],x[...,40:-40,40:-40])
                inner=F.mse_loss(fitted,target)
                latent_opt.zero_grad(set_to_none=True);inner.backward();latent_opt.step()
                with torch.no_grad():latent.clamp_(-128,127)
            auxiliary=F.mse_loss(nodes['res3_add']/16,latent.detach()/16)
            loss=loss+.25*auxiliary;values['refined_latent']=auxiliary
        if self.perception is not None:
            # Keep native720 forward/backward, but bound training-only VGG RAM.
            pp,tt=p,t
            if pp.shape[-1]>256 or pp.shape[-2]>256:
                pp,tt=pp[...,:256,:256],tt[...,:256,:256]
            pf=self.perception.extract(pp)
            with torch.no_grad():tf=self.perception.extract(tt)
            perceptual=torch.stack([F.l1_loss(pf[k],tf[k])/(tf[k].abs().mean()+1) for k in pf]).mean()
            loss=loss+self.args.perceptual*perceptual;values['perceptual']=perceptual
        if self.args.aux:
            aux=loss.new_zeros(())
            for key,node,scale in [('s2','down1.dw',2),('s4','down2.dw',4)]:
                target=F.pixel_unshuffle(x,scale)
                est=self.heads[key](nodes[node]/64-1)
                margin=40//scale
                target=target[...,margin:-margin,margin:-margin]
                est=est[...,margin:-margin,margin:-margin]
                # Supervise each sampling phase, including deviations from its
                # patch mean, rather than just low-frequency RGB averages.
                n,_,h,w=target.shape
                tr=target.reshape(n,3,scale*scale,h,w)
                er=est.reshape_as(tr)
                aux=aux+F.mse_loss(est,target)+F.mse_loss(er-er.mean(2,keepdim=True),tr-tr.mean(2,keepdim=True))
            loss=loss+self.args.aux*aux/2;values['aux']=aux/2
        self.opt.zero_grad(set_to_none=True);loss.backward()
        all_params=list(self.net.parameters())+list(self.heads.parameters())
        gn=torch.nn.utils.clip_grad_norm_(all_params,5,error_if_nonfinite=True)
        assert torch.isfinite(loss) and gn>0
        factor=.15+.85*.5*(1+math.cos(math.pi*min(self.step/self.args.steps,1)))
        for g,lr in zip(self.opt.param_groups,self.base_lrs):g['lr']=lr*factor
        self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),values={k:float(v.detach()) for k,v in values.items()},
                    grad=float(gn),native720=native,samples=specs,seconds=time.perf_counter()-start)

    def save(self,path):
        contract_guard()
        model=copy.deepcopy(self.net.state_dict());raw_scales={}
        if self.args.scale_lr:
            for key,param in self.net._parameters.items():
                if key.endswith('_m'):
                    raw_scales[key]=param.detach().clone()
                    model[key]=getattr(self.net,key).detach().clone()
        # model is directly loadable by the unchanged deployment class. Raw
        # fractional optimization state is separate and only used on resume.
        torch.save(dict(model=model,raw_scales=raw_scales,heads=self.heads.state_dict(),optimizer=self.opt.state_dict(),
            step=self.step,args=vars(self.args),rng=self.rng.getstate(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),contract_sha256=r.CONTRACT_SHA,
            residual_encoding=copy.deepcopy(self.net.package['residual_encoding'])),path)

    def restore(self,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        assert d['contract_sha256']==file_hash(r.ROOT/'network_ab/model_b_contract.json')
        self.net.load_state_dict(d['model']);self.heads.load_state_dict(d['heads']);self.opt.load_state_dict(d['optimizer'])
        if 'residual_encoding' in d:
            self.net.package['residual_encoding']=copy.deepcopy(d['residual_encoding'])
        if self.args.scale_lr:
            with torch.no_grad():
                for key,value in d['raw_scales'].items():self.net._parameters[key].copy_(value)
        self.step=d['step'];self.rng.setstate(d['rng']);torch.set_rng_state(d['torch_rng']);torch.cuda.set_rng_state_all(d['cuda_rng'])


@torch.inference_mode()
def evaluate(net,rows):
    records=[]
    for row in rows:
        x=r.tensor(r.read(row['path']))[None];p=net(x)[0]
        m=r.metrics(p,x)
        m['ssim']=float(ssim_map(p,x).mean())
        py=p.mean(1,keepdim=True);xy=x.mean(1,keepdim=True)
        mask=torch.zeros_like(xy,dtype=torch.bool)
        for d in (-1,-2):
            gx=torch.diff(xy,dim=d);gp=torch.diff(py,dim=d)
            m['edge_energy_ratio_'+str(d)]=float(gp.abs().sum()/gx.abs().sum().clamp_min(1e-8))
        # Fidelity of high-pass components, never edge energy alone.
        ph=p-F.avg_pool2d(F.pad(p,(2,2,2,2),mode='reflect'),5,1)
        th=x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,1)
        m['highpass_mae']=float((ph-th).abs().mean()*128)
        records.append(dict(id=row['id'],**m))
    mean=sum(v['mse_rgb255'] for v in records)/len(records)
    return dict(records=records,pooled_psnr=10*math.log10(255**2/mean),mean_mse=mean,
                mean_ssim=sum(v['ssim'] for v in records)/len(records),
                mean_edge=sum(v['edge_mae'] for v in records)/len(records),
                mean_highpass=sum(v['highpass_mae'] for v in records)/len(records))


def score(v, mode):
    if mode=='mse':return v['mean_mse']/128**2
    return v['mean_mse']/128**2+.25*sum(z['mae'] for z in v['records'])/len(v['records'])/128+.15*v['mean_edge']/128+.1*(1-v['mean_ssim'])


def preflight(args,rows,dest):
    a=Experiment(args,rows);torch.cuda.reset_peak_memory_stats()
    full=a.update(True);crop=a.update(False);a.save(dest/'preflight.pt')
    expected=a.update(False);state=copy.deepcopy(a.net.state_dict());head=copy.deepcopy(a.heads.state_dict())
    b=Experiment(args,rows);b.restore(dest/'preflight.pt');actual=b.update(False)
    error=max(float((v-state[k]).abs().max()) for k,v in b.net.state_dict().items())
    for k,v in b.heads.state_dict().items():error=max(error,float((v-head[k]).abs().max()))
    assert error<1e-5 and expected['samples']==actual['samples']
    # Identical images must have zero pixel/frequency errors and unit SSIM.
    x=torch.rand(1,3,48,48,device='cuda')*2-1
    assert float(focal_frequency(x,x))==0 and abs(float(ssim_map(x,x).mean())-1)<1e-6
    changed=x.clone();changed[...,::2]+=0.1
    assert float(focal_frequency(changed,x))>0
    write_json(dest/'preflight.json',dict(status='PASS',native=full,crop=crop,resume_error=error,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),qat=args.qat))
    print('RESEARCH_PREFLIGHT_PASS',args.name,error,flush=True)
    del a,b;torch.cuda.empty_cache()


def run(args):
    configure_workspace();set_reproducible(args.seed);contract_guard()
    train,val=r.data();dest=OUT/args.name;dest.mkdir(parents=True,exist_ok=args.resume)
    if not args.resume:
        shutil.copy2(__file__,dest/Path(__file__).name)
        write_json(dest/'config.json',dict(**vars(args),contract_sha256=r.CONTRACT_SHA,
            code_sha256=file_hash(__file__),training_frames=len(train),training_source_videos=7,
            training_heads_not_exported=True,hardware_changed=False,original_image_target=True))
        preflight(args,train,dest)
        if args.preflight_only:return
    tr=Experiment(args,train)
    if args.resume:
        tr.restore(dest/'last.pt')
        write_json(dest/f'resume_{tr.step:06d}.json',dict(restored_step=tr.step,
            checkpoint_sha256=file_hash(dest/'last.pt'),resumed_code_sha256=file_hash(__file__),
            args=vars(args)))
        shutil.copy2(__file__,dest/f'resumed_source_{tr.step:06d}.py')
    initial=evaluate(tr.net,val);best=score(initial,args.loss);bestps=initial['pooled_psnr'];beststruct=score(initial,'combined')
    if not args.resume:
        tr.save(dest/'best.pt');tr.save(dest/'best_psnr.pt');tr.save(dest/'best_structure.pt')
        history=[dict(step=0,validation=initial)]
    else:
        history=json.loads((dest/'history.json').read_text())
        best=min(score(z['validation'],args.loss) for z in history)
        bestps=max(z['validation']['pooled_psnr'] for z in history)
        beststruct=min(score(z['validation'],'combined') for z in history)
    significant=best;flat=0;began=time.perf_counter()
    with (dest/'metrics.jsonl').open('a' if args.resume else 'x',encoding='utf-8',buffering=1) as log:
        while tr.step<args.steps:
            if tr.step%100==0:
                assert shutil.disk_usage(OUT).free>250*1024**2,'E: insufficient free space; checkpoint preserved'
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('RESEARCH_STEP',args.name,tr.step,round(row['loss'],6),round(row['seconds'],3),flush=True)
            if tr.step%args.interval==0 or tr.step==args.steps:
                e=evaluate(tr.net,val);s=score(e,args.loss);ss=score(e,'combined')
                history.append(dict(step=tr.step,validation=e,score=s))
                if s<best:best=s;tr.save(dest/'best.pt')
                if e['pooled_psnr']>bestps:bestps=e['pooled_psnr'];tr.save(dest/'best_psnr.pt')
                if ss<beststruct:beststruct=ss;tr.save(dest/'best_structure.pt')
                if s<significant*.997:significant=s;flat=0
                else:flat+=1
                tr.save(dest/'last.pt');write_json(dest/'history.json',history)
                write_json(dest/'progress.json',dict(step=tr.step,validation=e,best_psnr=bestps,flat=flat,seconds=time.perf_counter()-began))
                print('RESEARCH_EVAL',args.name,tr.step,round(e['pooled_psnr'],4),round(e['mean_ssim'],5),round(e['mean_edge'],4),'flat',flat,flush=True)
                if tr.step>=args.minimum and flat>=args.patience:break
    state=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);tr.net.load_state_dict(state['model'])
    r.preview(tr.net,val,dest/'preview')
    if args.qat:r.export_review(tr.net,val,dest/'package')
    write_json(dest/'completion.json',dict(executed_steps=tr.step,selected_step=state['step'],
        best_psnr=bestps,stop='plateau' if tr.step<args.steps else 'stage_budget',
        theoretical_upper_bound=False,elapsed_seconds=time.perf_counter()-began))
    print('RESEARCH_COMPLETE',args.name,tr.step,bestps,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--minimum',type=int,default=3500)
    ap.add_argument('--interval',type=int,default=500);ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--seed',type=int,default=2026100557)
    ap.add_argument('--loss',choices=['mse','frequency','structure','combined'],default='mse')
    ap.add_argument('--warm',default='');ap.add_argument('--init',choices=['identity','independent'],default='identity')
    ap.add_argument('--aux',type=float,default=0);ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--crop',type=int,default=256);ap.add_argument('--batch',type=int,default=3)
    ap.add_argument('--hard',action='store_true');ap.add_argument('--qat',action='store_true')
    ap.add_argument('--colour',action='store_true',help='Training-only RGB permutation; input is also the exact target')
    ap.add_argument('--rotate',action='store_true',help='Training-only right angle rotations of square crops')
    ap.add_argument('--perceptual',type=float,default=0,help='Frozen local VGG source-feature loss, never style/Gram')
    ap.add_argument('--latent-refine',type=int,default=0,help='Training-only latent optimization targets; zero extra inference ops')
    ap.add_argument('--scale-lr',type=float,default=0,help='Learn existing per-channel Q17 multipliers; forward remains integer')
    ap.add_argument('--shift',type=float,default=0,help='Training-only translation consistency with original-image fidelity retained')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--preflight-only',action='store_true')
    run(ap.parse_args())


if __name__=='__main__':main()
