"""One final RGB target, coherent boundaries and error-flat region interiors.

No separate hard luminance classifier/latent target competes with the actual
image. Geometry and colour both refer to the unchanged stylized RGB target.
"""
import torch
from torch.nn import functional as F
from .train_b1_luma_chroma_interfaces import chroma_objective
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_rim_tolerance import rim_terms


def luma_regions(p,t,labels):
    y=lambda z:(z[:,:1]+2*z[:,1:2]+z[:,2:3])/4
    delta=y(p)-y(t);pixel=F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=2/128)
    seam=torch.zeros_like(labels,dtype=torch.bool)
    seam[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1];seam[:,1:]|=labels[:,1:]!=labels[:,:-1]
    core=(F.max_pool2d(seam[:,None].float(),5,1,2)==0)[:,0];means=[];flats=[]
    for z,lab,ok in zip(delta[:,0],labels,core):
        if not ok.any():continue
        _,idx=torch.unique(lab[ok],return_inverse=True);count=torch.bincount(idx).float();v=z[ok]
        mu=v.new_zeros(len(count)).scatter_add(0,idx,v)/count
        error=F.smooth_l1_loss(v-mu[idx],torch.zeros_like(v),beta=2/128,reduction='none')
        per=v.new_zeros(len(count)).scatter_add(0,idx,error)/count;weight=count.sqrt();weight/=weight.sum()
        means.append((F.smooth_l1_loss(mu,torch.zeros_like(mu),beta=2/128,reduction='none')*weight).sum())
        flats.append((per*weight).sum())
    mean=torch.stack(means).mean() if means else pixel*0;flat=torch.stack(flats).mean() if flats else pixel*0
    return 4*pixel+4*mean+4*flat,dict(final_luma_pixel=pixel,final_luma_region_mean=mean,final_luma_region_error_variation=flat)


class FullRGBGeometryTraining:
    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        colour,v=chroma_objective(p,t,labels);luma,l=luma_regions(p,t,labels)
        profile,detail=reference_boundary_profile(p,t,labels);rim,_=rim_terms(p,t,labels,2)
        v.update(l);v.update(final_rgb_profile=profile,two_pixel_region_centred_bright_shoulder=rim)
        return colour+luma+4*profile+rim,v


def regression():
    labels=torch.zeros(1,64,96,dtype=torch.long);labels[...,48:]=1
    t=torch.full((1,3,64,96),-.5);t[...,48:]=.5
    assert luma_regions(t,t,labels)[0]==0
    for name,p in [('flat',t*0),('blur',F.avg_pool2d(F.pad(t,(4,4,4,4),mode='replicate'),9,1)),('noise',t+torch.randn_like(t)*.05)]:
        p=p.detach().requires_grad_();loss,_=luma_regions(p,t,labels);loss.backward();assert loss>0 and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
    print('FULL_RGB_GEOMETRY_PASS target0/flat/blur/noise/finite_gradient',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
