"""Learn accepted cartoon appearance with the unchanged, fully learned B1.

Restoration is initialization only. Every supervised tensor is a style target.
Auxiliary phase heads, loss filters and flow are discarded before deployment.
"""
import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from . import research_b1_reconstruction as s
from . import train_b1_reconstruction as r
from .prepare_b1_style_recovery import OUT
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .train_b1_distillation import read_flow
from .coherent_shared_model import warp
from .b1_adversarial_balance import actual_output_balance


def data():
    d=json.loads((OUT/'data.json').read_text(encoding='utf-8'))
    assert d['contract_sha256']==r.CONTRACT_SHA
    for row in d['training']+d['validation']:
        for key,sha in [('path','sha256'),('target','target_sha256')]:
            assert file_hash(row[key])==row[sha]
    for row in d['pairs']: assert file_hash(row['flow'])==row['flow_sha256']
    return d


def local_distribution(p,t):
    """Local sliced distributions of RGB and signed gradients, training only.

    Inspired by SWD, but using local RGB/gradient projections rather than the
    paper's VGG feature distributions. No sampling or quantizer in inference.
    """
    directions=p.new_tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,1],
                             [1,-1,0],[0,1,-1],[1,0,-1]])
    directions=F.normalize(directions,dim=1)
    def distance(a,b,size):
        h,w=a.shape[-2:];h=h//size*size;w=w//size*size
        def project(z):
            z=torch.einsum('kc,nchw->nkhw',directions,z[...,:h,:w])
            z=z.unfold(2,size,size).unfold(3,size,size).flatten(-2)
            return z.sort(-1).values
        return F.l1_loss(project(a),project(b))
    colour=[];edges=[]
    for scale in (1,2):
        a,b=(p,t) if scale==1 else (F.avg_pool2d(p,2),F.avg_pool2d(t,2))
        colour.append(distance(a,b,8))
        for dim in (-1,-2):edges.append(distance(torch.diff(a,dim=dim),torch.diff(b,dim=dim),8))
    return torch.stack(colour).mean(),torch.stack(edges).mean()


def appearance(p,t,mode):
    values=dict(mse=F.mse_loss(p,t),l1=F.l1_loss(p,t))
    loss=values['mse']+.5*values['l1']
    if mode=='direct':return loss,values
    if mode=='blockshape':
        # Tiny B1 sampling-phase errors must not make a soft average preferable
        # to a sharp contour displaced by one or two native720 pixels.
        def magnitude(z):
            dx=F.pad(torch.diff(z,dim=-1).abs().mean(1,keepdim=True),(0,1,0,0))
            dy=F.pad(torch.diff(z,dim=-2).abs().mean(1,keepdim=True),(0,0,0,1))
            return (dx+dy)/2
        gp,gt=magnitude(p),magnitude(t)
        qp,qt=[1-torch.exp(-(g/(3/128)).square()) for g in (gp,gt)]
        neighbourhood=lambda v:F.max_pool2d(v,5,1,2)
        extra=F.relu(qp-neighbourhood(qt)).mean()
        missing=F.relu(qt-neighbourhood(qp)).mean()
        density=F.relu(F.avg_pool2d(qp,8)-F.avg_pool2d(qt,8)).mean()
        mask=(gt>6/128).float()
        sharp=(F.relu(gt-neighbourhood(gp))*mask).sum()/mask.sum().clamp_min(1)
        pp,tt=p,t
        if p.shape[-1]>320:pp,tt=p[...,200:520,400:720],t[...,200:520,400:720]
        colour,edges=local_distribution(pp,tt)
        values.update(extra_contours=extra,missing_contours=missing,contour_density=density,
                      local_sharpness=sharp,local_colour_distribution=colour,local_edge_distribution=edges)
        return .4*loss+.8*colour+.6*edges+.4*(extra+missing)+.6*density+.6*sharp,values
    edge=[];flat=[];boundary=[]
    for scale in (1,2,4):
        a,b=(p,t) if scale==1 else (F.avg_pool2d(p,scale),F.avg_pool2d(t,scale))
        for dim in (-1,-2):
            gp,gt=torch.diff(a,dim=dim),torch.diff(b,dim=dim)
            edge.append((gp-gt).abs().mean())
            mask=(gt.abs().mean(1,keepdim=True)<2/128).float()
            flat.append((F.relu(gp.abs()-gt.abs()-1/128)*mask).sum()/(3*mask.sum()+1e-8))
            mask=(gt.abs().mean(1,keepdim=True)>8/128).float()
            boundary.append(((gp-gt).abs()*mask).sum()/(3*mask.sum()+1e-8))
    values.update(edge=torch.stack(edge).mean(),flat=torch.stack(flat).mean(),
                  boundary=torch.stack(boundary).mean(),ssim=1-s.ssim_map(p,t).mean())
    loss=loss+.65*values['edge']+.3*values['flat']+.15*values['boundary']+.15*values['ssim']
    if mode=='frequency':
        values['frequency']=s.focal_frequency(p,t);loss=loss+.35*values['frequency']
    if mode=='distribution':
        # Full native frame already contributes all base appearance terms.
        pp,tt=p,t
        if p.shape[-1]>320:pp,tt=p[...,200:520,400:720],t[...,200:520,400:720]
        colour,edges=local_distribution(pp,tt)
        values.update(local_colour_distribution=colour,local_edge_distribution=edges)
        loss=loss+.7*colour+1.2*edges
    return loss,values


class ConditionalCritic(torch.nn.Module):
    """Two spatial scales, conditioned on coarse source structure; train only."""
    def __init__(self,bands=False):
        super().__init__();self.bands=bands;self.layers=torch.nn.ModuleList();ci=6
        for co,stride in [(24,1),(32,2),(48,2),(64,2),(1,1)]:
            self.layers.append(torch.nn.utils.spectral_norm(torch.nn.Conv2d(ci,co,3,stride,1)));ci=co

    def forward(self,image,source):
        results=[]
        def band_image(z):
            y=(z*z.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
            box=lambda size:F.avg_pool2d(F.pad(y,(size//2,)*4,mode='replicate'),size,1)
            b3,b9,b27=box(3),box(9),box(27)
            return torch.cat((4*(y-b3),2*(b3-b9),b9-b27),1)
        for scale in (1,2):
            a,b=(image,source) if scale==1 else (F.avg_pool2d(image,2),F.avg_pool2d(source,2))
            if self.bands:a,b=band_image(a),band_image(b)
            else:b=F.avg_pool2d(F.pad(b,(4,4,4,4),mode='replicate'),9,1)
            z=torch.cat((a,b),1)
            for i,layer in enumerate(self.layers):
                z=layer(z)
                if i<len(self.layers)-1:z=F.leaky_relu(z,.2)
            results.append(z)
        return results


class Training(s.Experiment):
    def __init__(self,args,d):
        super().__init__(args,d['training']);self.data=d
        if getattr(args,'regions',0):
            assert args.qat,'Ordered region head uses actual integer states'
            for key in ('up2_high_dw','rgb_head'):
                self.net.weight[key].requires_grad_(False);self.net.bias[key].requires_grad_(False)
        self.pair_groups={}
        for row in d['pairs']:self.pair_groups.setdefault(row['source'],[]).append(row)
        self.critic=None
        if getattr(args,'adversarial',0):
            self.critic=ConditionalCritic(getattr(args,'critic_bands',False)).cuda()
            self.dopt=torch.optim.Adam(self.critic.parameters(),lr=.00015,betas=(.5,.99))
            if getattr(args,'critic_warm',''):
                assert Path(args.critic_warm).resolve()==Path(args.warm).resolve()
                parent=torch.load(args.critic_warm,map_location='cpu',weights_only=True)
                self.critic.load_state_dict(parent['critic'])
                self.dopt.load_state_dict(parent['critic_optimizer'])
        self.qat_teacher=None
        if getattr(args,'qat_teacher',''):
            assert args.qat,'Float-output preservation is a quantization-only stage'
            state=torch.load(args.qat_teacher,map_location='cpu',weights_only=True)
            assert not state.get('qat',False) and not state.get('args',{}).get('qat',False)
            self.qat_teacher=r.initialize(args.seed)
            self.qat_teacher.load_state_dict(state['model']);self.qat_teacher.quantized=False
            self.qat_teacher.eval().requires_grad_(False)

    def save(self,path):
        super().save(path)
        if self.critic is not None:
            state=torch.load(path,map_location='cpu',weights_only=True)
            state.update(critic=self.critic.state_dict(),critic_optimizer=self.dopt.state_dict())
            torch.save(state,path)

    def restore(self,path):
        super().restore(path)
        if self.critic is not None:
            state=torch.load(path,map_location='cpu',weights_only=True)
            self.critic.load_state_dict(state['critic']);self.dopt.load_state_dict(state['critic_optimizer'])

    def batch_style(self,native):
        xs=[];ts=[];specs=[];flow=None;valid=None
        temporal=self.args.temporal>0 and not native and (self.step%4==0)
        if temporal:
            pair=self.rng.choice(self.pair_groups[self.rng.choice(sorted(self.pair_groups))])
            rows=[dict(id=Path(pair[k]).stem,path=pair[k],target=pair[k+'_target']) for k in ('current','next')]
        else:
            rows=[self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))]) for _ in range(1 if native else self.args.batch)]
        for i,row in enumerate(rows):
            a,b=r.read(row['path']),r.read(row['target'])
            if native:x=y=0;h,w=720,1280;flip=0
            elif not temporal or i==0:
                h=w=self.args.crop; choices=[]
                for _ in range(4 if self.args.hard and self.rng.randrange(2) else 1):
                    y=self.rng.randrange(721-h);x=self.rng.randrange(1281-w)
                    patch=b[y:y+h:4,x:x+w:4].astype(np.float32)
                    score=float(np.abs(np.diff(patch,axis=0)).mean()+np.abs(np.diff(patch,axis=1)).mean())
                    choices.append((score,x,y))
                _,x,y=max(choices);flip=0 if temporal else self.rng.randrange(2)
            aa,bb=a[y:y+h,x:x+w],b[y:y+h,x:x+w]
            if flip:aa=aa[:,::-1];bb=bb[:,::-1]
            xs.append(r.tensor(aa));ts.append(r.tensor(bb))
            specs.append(dict(id=row['id'],xy=[x,y],wh=[w,h],flip=flip))
        if temporal:
            f,v=read_flow(pair['flow']);f=f[y:y+h,x:x+w];v=v[y:y+h,x:x+w].copy()
            yy,xx=np.mgrid[:h,:w];m=40
            v*=((xx+f[:,:,0]>=m)&(xx+f[:,:,0]<w-m)&(yy+f[:,:,1]>=m)&(yy+f[:,:,1]<h-m))
            flow=torch.from_numpy(np.ascontiguousarray(f)).permute(2,0,1)[None].cuda()
            valid=torch.from_numpy(v)[None,None].cuda()
        return torch.stack(xs),torch.stack(ts),specs,flow,valid

    def update(self,native=None):
        began=time.perf_counter();native=(self.step+1)%150==0 if native is None else native
        x,t,specs,flow,valid=self.batch_style(native)
        if self.qat_teacher is not None:
            with torch.no_grad():t=self.qat_teacher(x)[0]
        nodes=self.net(x,all_nodes=True);pred=nodes['rgb_head']/128
        cut=lambda v:v if native else v[...,40:-40,40:-40]
        appearance_fn=getattr(self,'appearance_loss',appearance)
        loss,values=appearance_fn(cut(pred),cut(t),self.args.mode)
        if hasattr(self,'additional_loss'):
            extra,details=self.additional_loss(cut(pred),cut(t),native)
            loss+=extra;values.update(details)
        if hasattr(self,'input_stability_loss'):
            extra,details=self.input_stability_loss(x,pred,native)
            loss+=extra;values.update(details)
        if getattr(self.args,'regions',0):
            from .seed_b1_clear_regions import labels
            wanted=F.avg_pool2d(labels(t),2)
            feature=nodes['up2_low_pw'];margin=0 if native else 20
            if margin:feature=feature[...,margin:-margin,margin:-margin];wanted=wanted[...,margin:-margin,margin:-margin]
            logits=torch.stack([(feature[:,:6]-(j+.5)*127/3)/6 for j in range(3)],2)
            truth=torch.stack([(wanted[:,:6]>(j+.5)*127/3).float() for j in range(3)],2)
            ordinal=F.binary_cross_entropy_with_logits(logits,truth)
            chroma=F.l1_loss(feature[:,6:]/64,wanted[:,6:]/64)
            region=ordinal+.25*chroma+.1*F.l1_loss(feature[:,:6]/127,wanted[:,:6]/127)
            values.update(ordinal_regions=ordinal,region_chroma=chroma);loss+=self.args.regions*region
        if self.args.shift:
            dy,dx=self.rng.choice(((0,1),(1,0),(1,1),(2,2)))
            other=self.net(torch.roll(x[:1],(dy,dx),(-2,-1)))[0]
            aligned=torch.roll(other,(-dy,-dx),(-2,-1));m=80 if native else 40
            consistency=F.l1_loss(pred[:1,...,m:-m,m:-m],aligned[...,m:-m,m:-m])
            loss+=self.args.shift*consistency;values['phase']=consistency
        if flow is not None:
            warped=warp(pred[:1],flow);tw=warp(t[:1],flow)
            # Penalize actual variation where the target agrees across motion;
            # suppress moving boundaries, occlusions and changing target labels.
            if getattr(self.args,'temporal_mask','target')=='source':
                # Reliability must come from source correspondence. A jittering
                # offline target must not exempt its own unstable boundaries.
                agreement=(x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)
            else:
                agreement=(t[1:]-tw).abs().mean(1,keepdim=True)
            confidence=valid*torch.exp(-(agreement/.08)**2)
            q=cut((pred[1:]-warped).abs());v=cut(confidence)
            temporal=(q*v).sum()/(3*v.sum()+1e-8)
            loss+=self.args.temporal*temporal;values['temporal']=temporal
        if self.args.aux:
            aux=loss.new_zeros(())
            for key,node,scale in [('s2','down1.dw',2),('s4','down2.dw',4)]:
                target=F.pixel_unshuffle(t,scale);est=self.heads[key](nodes[node]/64-1)
                m=40//scale;target=target[...,m:-m,m:-m];est=est[...,m:-m,m:-m]
                n,_,h,w=target.shape;tr=target.reshape(n,3,scale*scale,h,w);er=est.reshape_as(tr)
                aux+=F.mse_loss(est,target)+F.mse_loss(er-er.mean(2,keepdim=True),tr-tr.mean(2,keepdim=True))
            values['aux']=aux/2;loss+=self.args.aux*aux/2
        if self.perception is not None:
            pp,tt=cut(pred),cut(t)
            if pp.shape[-1]>256 or pp.shape[-2]>256:
                pp,tt=pp[...,:256,:256],tt[...,:256,:256]
            pf=self.perception.extract(pp)
            with torch.no_grad():tf=self.perception.extract(tt)
            feature=torch.stack([F.l1_loss(pf[k],tf[k])/(tf[k].abs().mean()+1) for k in pf]).mean()
            values['target_features']=feature;loss+=self.args.perceptual*feature
        if self.critic is not None:
            pp,tt,xx=cut(pred),cut(t),cut(x)
            if native:
                yy=self.rng.randrange(0,720-256);xx0=self.rng.randrange(0,1280-256)
                pp,tt,xx=[z[...,yy:yy+256,xx0:xx0+256] for z in (pp,tt,xx)]
            self.critic.train().requires_grad_(True);self.dopt.zero_grad(set_to_none=True)
            real=self.critic(tt,xx);fake=self.critic(pp.detach(),xx)
            # Negative examples explicitly include lost boundaries and original
            # photographic texture, not just the current generator output.
            blurred=F.avg_pool2d(F.pad(tt,(2,2,2,2),mode='replicate'),5,1)
            soft=self.critic(blurred,xx);photo=self.critic(xx,xx)
            if getattr(self.args,'critic_artifacts',False):
                # Explicitly reject isolated bright/dark specks as well as
                # blur and photographic texture. Training-only negatives.
                seeds=(torch.rand_like(tt[:,:1])<.006).float()
                specks=F.max_pool2d(seeds,3,1,1)
                polarity=torch.where(torch.rand_like(tt[:,:1])<.5,-1.,1.)
                corrupted=(tt+.18*specks*polarity).clamp(-1,127/128)
                artifact=self.critic(corrupted,xx)
                dl=torch.stack([F.relu(1-a).mean()+.4*F.relu(1+b).mean()+
                                .2*(F.relu(1+c).mean()+F.relu(1+e).mean()+F.relu(1+f).mean())
                                for a,b,c,e,f in zip(real,fake,soft,photo,artifact)]).mean()
            else:
                dl=torch.stack([F.relu(1-a).mean()+.5*F.relu(1+b).mean()+
                                .25*F.relu(1+c).mean()+.25*F.relu(1+e).mean()
                                for a,b,c,e in zip(real,fake,soft,photo)]).mean()
            dl.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),5,error_if_nonfinite=True);self.dopt.step()
            self.critic.eval().requires_grad_(False)
            # Optional training-only backward guard. Its forward is required
            # to be identical, and discriminator training above is unchanged.
            critic_input=self.adversarial_input(pp,tt) if hasattr(self,'adversarial_input') else pp
            adv=-torch.stack([z.mean() for z in self.critic(critic_input,xx)]).mean()
            # Compare gradients on the same native-pixel crop, preventing an
            # arbitrary critic scale from overwhelming target fidelity.
            if getattr(self.args,'critic_balance','appearance')=='actual_output':
                factor,balance=actual_output_balance(loss,pred,adv,pp,self.args.adversarial,self.step/400)
                values.update(balance)
            else:
                reference=appearance(pp,tt,self.args.mode)[0]
                gp=torch.autograd.grad(reference,pp,retain_graph=True)[0]
                ga=torch.autograd.grad(adv,pp,retain_graph=True)[0]
                ratio=(gp.norm()/ga.norm().clamp_min(1e-8)).detach().clamp(.001,1.)
                factor=self.args.adversarial*ratio*min(1.,self.step/400)
            loss+=factor*adv;values.update(critic=dl.detach(),adversarial=adv,adversarial_factor=factor)
        self.opt.zero_grad(set_to_none=True);loss.backward()
        # Some training-only parameterizations transform gradients before the
        # shared norm clip. Ordinary optimizers retain their existing path.
        if hasattr(self.opt,'prepare_gradients'):self.opt.prepare_gradients()
        gn=torch.nn.utils.clip_grad_norm_(list(self.net.parameters())+list(self.heads.parameters()),5,error_if_nonfinite=True)
        assert torch.isfinite(loss) and gn>0
        factor=.2+.8*.5*(1+math.cos(math.pi*min(self.step/self.args.steps,1)))
        if getattr(self.args,'lr_schedule','cosine')=='constant':factor=1.
        for g,lr in zip(self.opt.param_groups,self.base_lrs):g['lr']=lr*factor
        self.opt.step()
        if getattr(self.args,'regions',0) and getattr(self.args,'tied_regions',False):
            # All six cumulative bands describe one luminance coordinate.
            # Project training weights onto that shared affine parameterization;
            # inference still reads six ordinary existing coefficient rows.
            from .seed_b1_clear_regions import BASES,INCREMENTS
            with torch.no_grad():
                w=self.net.weight['up2_low_pw'];b=self.net.bias['up2_low_pw']
                scale=w.new_tensor(INCREMENTS)/127;offset=b.new_tensor(BASES)
                common_w=(w[:6]*scale[:,None,None,None]).mean(0)
                common_b=(b[:6]*scale+offset).mean()
                w[:6].copy_(common_w[None]/scale[:,None,None,None])
                b[:6].copy_((common_b-offset)/scale)
        self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),values={k:float(v.detach()) for k,v in values.items()},
            grad=float(gn),native720=native,samples=specs,temporal_pair=flow is not None,seconds=time.perf_counter()-began)


@torch.inference_mode()
def evaluate(net,rows,teacher=None):
    records=[]
    for row in rows:
        x=r.tensor(r.read(row['path']))[None]
        t=r.tensor(r.read(row['target']))[None] if teacher is None else teacher(x)[0]
        p=net(x)[0]
        loss,v=appearance(p,t,'structured');metrics=r.metrics(p,t)
        metrics.update({k:float(z) for k,z in v.items()})
        metrics['ssim']=1-metrics['ssim'];metrics['score']=float(loss)
        # Lower residual source detail is only good when target edge fidelity is retained.
        ep=sum(torch.diff(p,dim=d).abs().sum() for d in (-1,-2))
        et=sum(torch.diff(t,dim=d).abs().sum() for d in (-1,-2))
        metrics['edge_energy_ratio']=float(ep/et.clamp_min(1e-8))
        metrics['source_mae']=float((p-x).abs().mean()*128)
        records.append(dict(id=row['id'],**metrics))
    mean={k:sum(v[k] for v in records)/len(records) for k in records[0] if k!='id'}
    mean['pooled_psnr']=10*math.log10(255**2/mean['mse_rgb255'])
    return dict(records=records,mean=mean)


@torch.inference_mode()
def preview(net,rows,dest):
    dest.mkdir(exist_ok=True);ims={}
    for row in rows:
        x=r.tensor(r.read(row['path']))[None];p=net(x)[0]
        output=r.c.rgb_image(p[0]);output.save(dest/(row['id']+'_output.png'))
        ims[row['id']]=[Image.fromarray(r.read(row['path'])),Image.fromarray(r.read(row['target'])),output]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1260,314*len(r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(r.c.CROPS):
        for j,(im,title) in enumerate(zip(ims[key],['原图','本轮训练目标（离线）','实际B1网络输出'])):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')


def preflight(args,d,dest):
    a=Training(args,d);torch.cuda.reset_peak_memory_stats()
    if args.adversarial:a.step=400  # Also test the fully enabled adversarial term.
    native=a.update(True);crop=a.update(False)
    # Resume exactly across a real adjacent-flow step when temporal loss is on.
    a.update(False);a.update(False);a.save(dest/'preflight.pt')
    expected=a.update(False);state=copy.deepcopy(a.net.state_dict());head=copy.deepcopy(a.heads.state_dict())
    critic=copy.deepcopy(a.critic.state_dict()) if a.critic is not None else None
    b=Training(args,d);b.restore(dest/'preflight.pt');actual=b.update(False)
    err=max(float((v-state[k]).abs().max()) for k,v in b.net.state_dict().items())
    for k,v in b.heads.state_dict().items():err=max(err,float((v-head[k]).abs().max()))
    if critic is not None:
        for k,v in b.critic.state_dict().items():err=max(err,float((v-critic[k]).abs().max()))
    assert err<1e-5 and expected['samples']==actual['samples']
    z=torch.rand(1,3,64,64,device='cuda')*2-1
    l,_=appearance(z,z,args.mode);assert abs(float(l))<1e-6
    l2,_=appearance(z.roll(1,-1),z,args.mode);assert l2>0
    write_json(dest/'preflight.json',dict(status='PASS',native=native,crop=crop,replayed=actual,resume_error=err,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),qat=args.qat,loss_identity_zero=True))
    print('STYLE_PREFLIGHT_PASS',args.name,err,flush=True)
    del a,b;torch.cuda.empty_cache()


def run(args):
    configure_workspace();set_reproducible(args.seed);cv2.setNumThreads(2);contract_guard()
    d=data();dest=OUT/args.name;dest.mkdir(exist_ok=args.resume)
    if not args.resume:
        shutil.copy2(__file__,dest/Path(__file__).name)
        paths=[Path(__file__),Path(s.__file__),Path(r.__file__),Path(r.c.__file__)]
        paths += [Path(__file__).with_name(n) for n in ('losses.py','coherent_shared_model.py','train_b1_distillation.py','engine.py')]
        if args.regions:
            helper=Path(__file__).with_name('seed_b1_clear_regions.py');paths.append(helper)
            shutil.copy2(helper,dest/helper.name)
        write_json(dest/'config.json',dict(args=vars(args),data_sha256=file_hash(OUT/'data.json'),
            code={str(p):file_hash(p) for p in paths},warm_sha256=file_hash(args.warm) if args.warm else None,
            original_identity_loss=False,hardware_changed=False,precision='FP32_without_AMP_or_TF32',
            contract_sha256=r.CONTRACT_SHA,
            existing_vgg_sha256=file_hash(r.ROOT/'model-training-workspace/cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth') if args.perceptual else None,
            qat_teacher_sha256=file_hash(args.qat_teacher) if args.qat_teacher else None,
            critic_warm_sha256=file_hash(args.critic_warm) if args.critic_warm else None))
        preflight(args,d,dest)
        if args.preflight_only:return
    tr=Training(args,d)
    def assess():
        e=evaluate(tr.net,d['validation'])
        if tr.qat_teacher is not None:e['fp32_preservation']=evaluate(tr.net,d['validation'],tr.qat_teacher)
        return e
    def rank(e):return e.get('fp32_preservation',e)['mean']['score']
    if args.resume:
        tr.restore(dest/'last.pt');history=json.loads((dest/'history.json').read_text())
        best=min(rank(v['validation']) for v in history)
    else:
        e=assess();history=[dict(step=0,validation=e)]
        best=rank(e);tr.save(dest/'best.pt')
    flat=0;significant=best;start=time.perf_counter();stop_reason=None
    with (dest/'metrics.jsonl').open('a' if args.resume else 'x',encoding='utf-8',buffering=1) as log:
        while tr.step<args.steps:
            if tr.step%100==0:assert shutil.disk_usage(OUT).free>512*1024**2,'Disk guard'
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('STYLE_STEP',args.name,tr.step,round(row['loss'],5),round(row['seconds'],3),flush=True)
            if tr.step%args.interval==0 or tr.step==args.steps:
                e=assess();q=rank(e);history.append(dict(step=tr.step,validation=e))
                if q<best:best=q;tr.save(dest/'best.pt');preview(tr.net,d['validation'],dest/'preview')
                if args.adversarial or args.mode=='blockshape' or args.regions:
                    # Per-pixel ranking may prefer a photographic compromise.
                    # Preserve actual candidates for explicit visual selection.
                    tr.save(dest/f'candidate_{tr.step:06d}.pt')
                    preview(tr.net,d['validation'],dest/'latest-preview')
                if q<significant*.997:significant=q;flat=0
                else:flat+=1
                tr.save(dest/'last.pt');write_json(dest/'history.json',history)
                write_json(dest/'progress.json',dict(step=tr.step,validation=e,best_score=best,flat=flat,seconds=time.perf_counter()-start))
                print('STYLE_EVAL',args.name,tr.step,{k:round(e['mean'][k],5) for k in ('pooled_psnr','ssim','edge_energy_ratio','score')},flush=True)
                request=dest/'STOP_REQUEST.json'
                if request.exists():
                    stop_reason=json.loads(request.read_text(encoding='utf-8'))
                    assert stop_reason.get('reason'),'Cooperative stop must record evidence'
                    break
                if tr.step>=args.minimum and flat>=args.patience:break
    beststate=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);tr.net.load_state_dict(beststate['model'])
    preview(tr.net,d['validation'],dest/'preview')
    write_json(dest/'completion.json',dict(executed_steps=tr.step,selected_step=beststate['step'],
        elapsed_seconds=time.perf_counter()-start,validation=assess(),
        ranking='selected_FP32_preservation' if tr.qat_teacher is not None else 'accepted_style_error',
        stop='reviewed_stop' if stop_reason else ('plateau' if tr.step<args.steps else 'stage_budget'),
        stop_evidence=stop_reason,globally_optimal=False))
    print('STYLE_COMPLETE',args.name,tr.step,beststate['step'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--minimum',type=int,default=3500)
    ap.add_argument('--interval',type=int,default=500);ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--seed',type=int,default=2026100581)
    ap.add_argument('--mode',choices=['direct','structured','frequency','distribution','blockshape'],default='structured')
    ap.add_argument('--warm',default=str(s.OUT/'long-tail-fp32/best_psnr.pt'))
    ap.add_argument('--init',default='identity');ap.add_argument('--aux',type=float,default=0)
    ap.add_argument('--lr',type=float,default=.0003);ap.add_argument('--crop',type=int,default=256)
    ap.add_argument('--batch',type=int,default=3);ap.add_argument('--hard',action='store_true')
    ap.add_argument('--qat',action='store_true');ap.add_argument('--scale-lr',type=float,default=0)
    ap.add_argument('--shift',type=float,default=0);ap.add_argument('--temporal',type=float,default=0)
    ap.add_argument('--perceptual',type=float,default=0,help='Frozen existing VGG target-style feature matching')
    ap.add_argument('--adversarial',type=float,default=0,help='Train-only conditional edge critic gradient ratio')
    ap.add_argument('--critic-bands',action='store_true',help='Remove global colour-offset shortcut from train-only critic')
    ap.add_argument('--critic-warm',default='',help='Continue an existing critic alongside the same generator warm checkpoint')
    ap.add_argument('--regions',type=float,default=0,help='Ordered luminance/chroma supervision with the verified fixed integer head')
    ap.add_argument('--tied-regions',action='store_true',help='Project six PW rows to one shared affine luminance coordinate')
    ap.add_argument('--qat-teacher',default='',help='Frozen chosen FP32 output as QAT fidelity target; validation still uses accepted style')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--preflight-only',action='store_true')
    ap.set_defaults(colour=False,rotate=False,latent_refine=0)
    run(ap.parse_args())


if __name__=='__main__':main()
