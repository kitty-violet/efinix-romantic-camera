"""Local projected RGB distributions across actual target boundaries.

Inspired by sliced Wasserstein losses (Heitz et al., CVPR2021), but this is
a fixed 13-direction, local RGB W1 variant, not their feature-texture model.
Shared +/-2px shift is permitted; full target AA and colour populations stay.
Position/order are still supervised by the separate complete-profile term.
"""
import itertools
import torch
from torch.nn import functional as F


def directions(device,dtype):
    ds=[v for v in itertools.product((-1,0,1),repeat=3) if any(v) and next(x for x in v if x)!=-1]
    v=torch.tensor(ds,device=device,dtype=dtype)
    return v/v.norm(dim=1,keepdim=True)


def palette_distribution(p,t,labels,max_samples=1024):
    results=[];counts=[];ds=directions(p.device,p.dtype)
    for axis in (-1,-2):
        pp=p if axis==-1 else p.transpose(-1,-2)
        tt=t if axis==-1 else t.transpose(-1,-2)
        ll=labels if axis==-1 else labels.transpose(-1,-2)
        mask=(ll[...,1:]!=ll[...,:-1]) & (torch.diff(tt,dim=-1).abs().amax(1)>4/128)
        mask[...,:8]=False;mask[...,-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
        idx=mask.nonzero()
        if len(idx)>max_samples:idx=idx[torch.linspace(0,len(idx)-1,max_samples,device=p.device).long()]
        if not len(idx):continue
        bb,yy,xx=idx.unbind(1);offset=torch.arange(-4,6,device=p.device)
        def gather(v,shift):return v.permute(0,2,3,1)[bb[:,None],yy[:,None],xx[:,None]+offset+shift]
        ref=(gather(tt,0)@ds.T).sort(dim=1).values
        choices=[]
        for shift in (-2,-1,0,1,2):
            projected=(gather(pp,shift)@ds.T).sort(dim=1).values
            choices.append((projected-ref).abs().mean((1,2)))
        results.append(torch.stack(choices).amin(0).mean());counts.append(len(idx))
    value=torch.stack(results).mean() if results else p.sum()*0
    return value,dict(boundary_palette_distribution=value,palette_distribution_profiles=p.new_tensor(sum(counts)))


def regression():
    assert directions('cpu',torch.float32).shape==(13,3)
    lab=torch.zeros(1,64,96,dtype=torch.long);lab[...,48:]=1
    t=torch.full((1,3,64,96),-.6);t[...,48:]=.6
    assert palette_distribution(t,t,lab)[0]==0
    assert palette_distribution(t.roll(2,-1),t,lab)[0]<1e-7
    aa=F.avg_pool2d(F.pad(t,(1,1,1,1),mode='replicate'),3,1)
    assert palette_distribution(aa,aa,lab)[0]==0
    p=F.avg_pool2d(F.pad(t,(4,4,4,4),mode='replicate'),9,1).requires_grad_()
    loss,_=palette_distribution(p,t,lab);loss.backward()
    assert loss>0 and torch.isfinite(p.grad).all() and (p.grad*(t-p.detach())).sum()<0
    for width in (1,2,3,4):
        labels=torch.zeros_like(lab);labels[...,48:48+width]=1
        target=torch.full_like(t,-.6);target[...,48:48+width]=.6
        assert palette_distribution(target*0-.6,target,labels)[0]>0
    # Same per-channel histograms are insufficient: mixed RGB projections
    # distinguish red/cyan from blue/yellow joint colours.
    target=torch.zeros_like(t);target[:,0,:,:48]=.8;target[:,1:, :,48:]=.8
    wrong=target[:,[2,1,0]].clone()
    assert palette_distribution(wrong,target,lab)[0]>.01
    assert palette_distribution(t,t,lab*0)[0]==0
    print('BOUNDARY_PALETTE_PASS target/AA/shift2/blur_gradient/thin_regions/joint_RGB/empty',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
