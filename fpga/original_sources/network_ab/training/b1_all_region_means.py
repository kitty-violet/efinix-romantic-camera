"""Region mean errors include narrow regions; flatness remains core-only.

All comparisons use prediction-minus-AA-target, so the existing target is
still an exact zero. This is a training objective, not a runtime filter.
"""
import torch
from torch.nn import functional as F
from .train_b1_luma_chroma_interfaces import chroma


def core_mask(labels):
    seam=torch.zeros_like(labels,dtype=torch.bool)
    seam[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    seam[:,1:]|=labels[:,1:]!=labels[:,:-1]
    return (F.max_pool2d(seam[:,None].float(),5,1,2)==0)[:,0]


def error_terms(delta,labels):
    assert delta.shape[0]==labels.shape[0] and delta.shape[-2:]==labels.shape[-2:]
    pixel=F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=2/128)
    means=[];flats=[]
    for d,lab,ok in zip(delta,labels,core_mask(labels)):
        _,idx=torch.unique(lab.flatten(),return_inverse=True)
        count=torch.bincount(idx).to(d.dtype);v=d.flatten(1)
        mu=d.new_zeros(d.shape[0],len(count)).scatter_add(1,idx[None].expand_as(v),v)/count
        weight=count.sqrt();weight=weight/weight.sum()
        means.append((F.smooth_l1_loss(mu,torch.zeros_like(mu),beta=2/128,reduction='none').mean(0)*weight).sum())
        # Keep the prior error-flatness definition exactly. Boundary pixels
        # are not treated as flat interiors, even though they now affect mean.
        if not ok.any():continue
        _,ci=torch.unique(lab[ok],return_inverse=True);cv=d[:,ok]
        cc=torch.bincount(ci).to(d.dtype)
        cm=d.new_zeros(d.shape[0],len(cc)).scatter_add(1,ci[None].expand_as(cv),cv)/cc
        err=F.smooth_l1_loss(cv-cm[:,ci],torch.zeros_like(cv),beta=2/128,reduction='none').mean(0)
        per=d.new_zeros(len(cc)).scatter_add(0,ci,err)/cc;cw=cc.sqrt();cw=cw/cw.sum()
        flats.append((per*cw).sum())
    return pixel,torch.stack(means).mean(),torch.stack(flats).mean() if flats else pixel*0


def all_region_colour_objective(p,t,labels):
    y=lambda z:(z[:,:1]+2*z[:,1:2]+z[:,2:3])/4
    yp,ym,yf=error_terms(y(p)-y(t),labels)
    cp,cm,cf=error_terms(chroma(p)-chroma(t),labels)
    return 4*(yp+ym+yf)+8*cp+4*(cm+cf),dict(final_luma_pixel=yp,
        all_region_luma_mean=ym,final_luma_region_error_variation=yf,
        chroma_pixel=cp,all_region_chroma_mean=cm,chroma_region_residual_variation=cf)


def regression():
    from .train_b1_full_rgb_geometry import luma_regions
    from .train_b1_luma_chroma_interfaces import chroma_objective
    records=[]
    for width in (1,2,3,4,6,12):
        lab=torch.zeros(1,40,64,dtype=torch.long);lab[...,30:30+width]=1
        t=torch.full((1,3,40,64),-.5);t[:,0,:,30:30+width]=.7;t[:,1,:,30:30+width]=.3
        p=torch.full_like(t,-.5).requires_grad_()
        _,yv=luma_regions(p,t,lab);_,cv=chroma_objective(p,t,lab)
        new,v=all_region_colour_objective(p,t,lab)
        assert v['all_region_luma_mean']>0 and v['all_region_chroma_mean']>0
        assert torch.allclose(v['final_luma_region_error_variation'],yv['final_luma_region_error_variation'],atol=1e-7)
        assert torch.allclose(v['chroma_region_residual_variation'],cv['chroma_region_residual_variation'],atol=1e-7)
        if width<=4:assert yv['final_luma_region_mean']==0 and cv['chroma_region_mean']==0
        new.backward();assert torch.isfinite(p.grad).all() and (p.grad*(t-p.detach())).sum()<0
        for target in (t,F.avg_pool2d(F.pad(t,(1,1,1,1),mode='replicate'),3,1)):
            assert abs(float(all_region_colour_objective(target,target,lab)[0]))<1e-10
        records.append(dict(width=width,old_y_mean=float(yv['final_luma_region_mean'].detach()),new_y_mean=float(v['all_region_luma_mean'].detach())))
    print('ALL_REGION_MEAN_PASS',records,flush=True)
    return records


if __name__=='__main__':torch.set_num_threads(2);regression()
