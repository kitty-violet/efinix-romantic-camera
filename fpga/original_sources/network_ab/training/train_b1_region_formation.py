"""Whole-region interiors plus sharp, position-tolerant boundaries.

A concave gradient cost favours concentrated transitions over diffuse ramps.
Unlike exact-position gradient regression, a clean contour displaced3px does
not receive a worse boundary score than a9px blurred transition.
"""
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from . import train_b1_target_latent as parent
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m
from .runtime import write_json


def formation_loss(p,target,labels):
    boundary=torch.zeros_like(labels,dtype=torch.bool)
    boundary[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    boundary[:,1:]|=labels[:,1:]!=labels[:,:-1]
    mask=(F.max_pool2d(boundary[:,None].float(),9,1,4)==0).float()
    flats=[];means=[]
    for pp,tt,ll,mm in zip(p,target,labels,mask):
        index=ll.flatten().long();n=int(index.max())+1;ids=index[None].expand(3,-1);valid=mm.flatten()
        count=valid.new_zeros(n).scatter_add(0,index,valid).clamp_min(1)
        mu=pp.new_zeros(3,n).scatter_add(1,ids,pp.flatten(1)*valid)/count[None]
        mt=tt.new_zeros(3,n).scatter_add(1,ids,tt.flatten(1)*valid)/count[None]
        flats.append(((pp.flatten(1)-mu[:,index]).abs()*valid).sum()/(3*valid.sum()).clamp_min(1))
        means.append(((mu[:,index]-mt[:,index]).abs()*valid).sum()/(3*valid.sum()).clamp_min(1))
    flat=torch.stack(flats).mean();colour=torch.stack(means).mean();missing=[];sp=[];st=[]
    eps=.5/128
    for axis in (-1,-2):
        gp,gt=torch.diff(p,dim=axis),torch.diff(target,dim=axis)
        sp.append(((gp.abs().mean(1,keepdim=True)+eps).sqrt()-eps**.5).mean())
        st.append(((gt.abs().mean(1,keepdim=True)+eps).sqrt()-eps**.5).mean())
        for sign in (-1,1):
            a,b=(gp*sign).clamp_min(0),(gt*sign).clamp_min(0)
            near=F.max_pool2d(a,9,1,4);important=(b>6/128).float()
            missing.append((F.relu(b-near)*important).sum()/important.sum().clamp_min(1))
    edge=torch.stack(missing).mean()
    sparse=torch.stack(sp).mean();target_sparse=torch.stack(st).mean()
    excess=F.relu(sparse-1.05*target_sparse)
    loss=2.5*flat+.8*colour+2*edge+4*excess
    return loss,dict(region_interior_flat=flat,region_interior_colour=colour,
        missing_sharp_boundary=edge,excess_diffuse_transitions=excess,gradient_cost=sparse,target_gradient_cost=target_sparse)


class FormationTraining(parent.TargetLatentTraining):
    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        area,values=formation_loss(p,target,labels)
        with torch.no_grad():wanted=self.teacher(self.full_target,all_nodes=True)['res3_add']
        actual=self.nodes['res3_add']
        if not native:actual=actual[...,10:-10,10:-10];wanted=wanted[...,10:-10,10:-10]
        scale=wanted.std((0,2,3),keepdim=True).clamp_min(2).detach()
        code=F.smooth_l1_loss(actual/scale,wanted/scale)
        values['target_latent']=code
        return self.args.area_weight*area+self.args.classification*code,values


def unit_check():
    labels=torch.zeros(1,128,128,dtype=torch.long,device='cuda');labels[:,:,64:]=1
    target=torch.where(labels[:,None]==0,-.3,.3).expand(-1,3,-1,-1).contiguous()
    shifted=target.clone();shifted[:,:,:,64:67]=-.3
    blurred=F.avg_pool2d(F.pad(target,(4,4,4,4),mode='replicate'),9,1)
    generator=torch.Generator(device='cuda').manual_seed(13)
    noisy=target+torch.randn(target.shape,device='cuda',generator=generator)*.1
    cases={'exact':target,'sharp_shift3':shifted,'blur9':blurred,'constant':target*0,'noise':noisy}
    old={k:float(m.region_loss(p,target,labels)[0]) for k,p in cases.items()}
    new={k:float(formation_loss(p,target,labels)[0]) for k,p in cases.items()}
    print('FORMATION_LOSS_VALUES',old,new,flush=True)
    assert old['blur9']<old['sharp_shift3']
    # FP32 scatter accumulation of thousands of identical .3 values has a
    # measured nonzero roundoff floor; the rank test remains orders larger.
    assert new['exact']<1e-4 and new['sharp_shift3']<1e-4
    assert min(new[k] for k in ('blur9','constant','noise'))>.1
    path=m.OUT/'formation-loss-regression.json'
    if not path.exists():write_json(path,dict(old=old,new=new,scope='Synthetic loss regression; not visual acceptance'))
    print('FORMATION_LOSS_REGRESSION',old,new,flush=True)


@torch.inference_mode()
def evaluate(net,rows):
    records=[]
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        p=net(x)[0];labels=torch.from_numpy(m.load_region(row['region']).astype('int64'))[None].cuda()
        loss,values=formation_loss(p,target,labels);appearance,_=m.t.appearance(p,target,'direct')
        metrics=m.t.r.metrics(p,target);metrics.update({k:float(v) for k,v in values.items()})
        metrics['score']=float(loss+.25*appearance);records.append(dict(id=row['id'],**metrics))
    mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'}
    return dict(records=records,mean=mean)


if __name__=='__main__':
    unit_check()
    driver.main(training_factory=FormationTraining,extra_sources=[Path(__file__),Path(parent.__file__)],evaluation=evaluate)
