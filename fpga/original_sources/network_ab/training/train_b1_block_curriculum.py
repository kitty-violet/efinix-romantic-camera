"""Teach clean block reproduction before removing real photographic texture.

Evaluation always uses original source images. Synthetic/curriculum inputs
exist only during training and are never displayed as inference achievements.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_region_formation as formation
from . import train_b1_late_region_supervision as driver
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_clean_regions import CleanRegionTraining
from .runtime import write_json


@lru_cache(maxsize=12)
def source_region_mean(path,region):
    rgb=m.t.r.read(path).astype(np.float32);labels=m.load_region(region);n=int(labels.max())+1
    count=np.bincount(labels.ravel(),minlength=n).clip(1)
    means=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel(),minlength=n)/count for c in range(3)],1)
    return (means[labels]/128-1).astype(np.float32)


def boundary_tolerant_appearance(p,target,mode):
    edge=torch.zeros_like(target[:,:1],dtype=torch.bool)
    edge[:,:,:,1:]|=torch.diff(target,dim=-1).abs().amax(1,keepdim=True)>3/128
    edge[:,:,1:]|=torch.diff(target,dim=-2).abs().amax(1,keepdim=True)>3/128
    interior=(F.max_pool2d(edge.float(),9,1,4)==0).float()
    flat=((p-target).abs()*interior).sum()/(3*interior.sum()).clamp_min(1)
    coarse=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(target,16))
    return .6*flat+.3*coarse,dict(interior_fidelity=flat,coarse_colour=coarse)


def block_objective(p,target,labels):
    base,values=formation.formation_loss(p,target,labels)
    # Do not let missing edges elsewhere buy an allowance for local noise.
    # A concave absolute gradient cost penalizes diffuse ramps and tiny islands.
    loss=base-4*values['excess_diffuse_transitions']+4*values['gradient_cost']
    return loss,values


class BlockCurriculum(m.AreaTraining):
    appearance_loss=staticmethod(boundary_tolerant_appearance)
    input_stability_loss=CleanRegionTraining.input_stability_loss

    def __init__(self,args,data):
        args.freeze_head=False
        super().__init__(args,data);state=torch.load(args.warm,map_location='cpu',weights_only=True)
        if 'residual_encoding' in state:self.net.package['residual_encoding']=state['residual_encoding']
        else:
            from .b1_trained_image import decode
            raw=(Path(args.warm).parent/'coefficients-padded.bin').read_bytes()
            _,self.net.package['residual_encoding']=decode(raw);assert self.net.export()==raw
        groups=[]
        for key,w in self.net.weight.items():
            w.requires_grad_(True);self.net.bias[key].requires_grad_(True)
            scale=(getattr(self.net,key+'_m')/131072).float().flatten()
            groups.append(dict(params=[w],lr=args.lr,row_gain=(scale/(1/256))[:,None,None,None]))
            groups.append(dict(params=[self.net.bias[key]],lr=.008))
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def batch_style(self,native):
        x,target,specs,flow,valid=super().batch_style(native)
        fraction=self.step/self.args.steps
        detail=max(0.,min(1.,(fraction-.10)/.20))
        real=max(0.,min(1.,(fraction-.30)/.30))
        if real<1:
            values=[]
            for spec in specs:
                row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
                v=source_region_mean(row['path'],row['region']);xx,yy=spec['xy'];w,h=spec['wh'];v=v[yy:yy+h,xx:xx+w]
                if spec['flip']:v=v[:,::-1]
                values.append(torch.from_numpy(np.ascontiguousarray(v)).permute(2,0,1).cuda())
            mean=torch.stack(values)
            textured=(target+detail*(x-mean)).clamp(-1,127/128)
            x=(1-real)*textured+real*x
            # Offline region maps may move inconsistently; do not teach their
            # synthetic motion as if it were measured real source correspondence.
            flow=valid=None
        self.curriculum=dict(detail_fraction=detail,real_source_fraction=real)
        return x,target,specs,flow,valid

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        loss,values=block_objective(p,target,labels)
        return loss,values

    def update(self,native=None):
        result=super().update(native);result['curriculum']=self.curriculum;return result

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(residual_encoding=self.net.package['residual_encoding'],block_curriculum=True)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    records=[]
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].cuda()
        p=net(x)[0];shape,values=block_objective(p,target,labels);colour,_=boundary_tolerant_appearance(p,target,'direct')
        record=dict(id=row['id'],**m.t.r.metrics(p,target),**{k:float(v) for k,v in values.items()},score=float(shape+colour))
        records.append(record)
    return dict(records=records,mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'})


def regression():
    labels=torch.zeros(1,128,128,device='cuda',dtype=torch.long);labels[:,:,64:]=1
    target=torch.where(labels[:,None]==0,-.3,.3).expand(-1,3,-1,-1).contiguous()
    shifted=target.clone();shifted[:,:,:,64:67]=-.3
    blurred=F.avg_pool2d(F.pad(target,(4,4,4,4),mode='replicate'),9,1)
    noisy=target+.1*torch.randn_like(target)
    values={k:float(block_objective(v,target,labels)[0]+boundary_tolerant_appearance(v,target,'direct')[0]) for k,v in
        [('target',target),('shift3',shifted),('blur9',blurred),('noise',noisy),('constant',target*0)]}
    assert values['target']<min(values[k] for k in ('blur9','noise','constant'))
    assert values['shift3']<values['blur9']
    print('BLOCK_CURRICULUM_LOSS_REGRESSION',values,flush=True)


if __name__=='__main__':
    regression()
    driver.main(training_factory=BlockCurriculum,extra_sources=[Path(__file__),Path(formation.__file__),
        Path(__file__).with_name('train_b1_quantum_regions.py'),Path(__file__).with_name('train_b1_clean_regions.py')],evaluation=evaluate)
