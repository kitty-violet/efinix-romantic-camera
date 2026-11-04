"""Learn source-texture invariance and motion-consistent colour-region labels.

Real TRAIN geometry supplies controlled texture interventions: region mean is
retained while interior contrast changes. This differs from the old synthetic
Voronoi patches. Adjacent-frame supervision uses an anchor's transported labels
only where existing source flow is reliable. No labels/flow/filters at inference.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_region_tail_flatness import RegionTailTraining,evaluate,preview
from .coherent_shared_model import warp
from . import train_b1_late_region_supervision as driver


def texture_intervention(x,labels,gains=None,rng=None):
    result=[]
    for image,lab in zip(x,labels):
        _,idx,count=lab.flatten().unique(return_inverse=True,return_counts=True)
        ids=idx[None].expand(3,-1)
        mean=image.new_zeros(3,len(count)).scatter_add(1,ids,image.flatten(1))/count[None]
        if gains is None:
            gain=image.new_tensor([rng.uniform(0,1.6) for _ in range(len(count))])
        else:gain=torch.full_like(count,float(gains),dtype=image.dtype)
        changed=mean[:,idx]+gain[idx][None]*(image.flatten(1)-mean[:,idx])
        result.append(changed.reshape_as(image).clamp(-1,127/128))
    return torch.stack(result)


def nearest_warp(labels,flow):
    n,h,w=labels.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=flow.device,dtype=flow.dtype),
                        torch.arange(w,device=flow.device,dtype=flow.dtype),indexing='ij')
    grid=torch.stack((2*(xx[None]+flow[:,0])/max(w-1,1)-1,
                      2*(yy[None]+flow[:,1])/max(h-1,1)-1),-1)
    return F.grid_sample(labels[:,None].float(),grid,mode='nearest',padding_mode='border',align_corners=True)[:,0].long()


@torch.no_grad()
def align_pair(x,target,labels,codes,flow,valid):
    disagreement=(x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)
    confidence=valid*torch.exp(-(disagreement/.08)**2)
    # Exclude the rim of uncertain/occluded areas, not colour boundaries.
    reliable=(confidence>.8).float()
    reliable=(-F.max_pool2d(-F.pad(reliable,(2,2,2,2),value=0),5,1)>0)[:,0]
    t=target.clone();ll=labels.clone();cc=codes.clone()
    t[1:]=torch.where(reliable[:,None],warp(target[:1],flow),target[1:])
    ll[1:]=torch.where(reliable,nearest_warp(labels[:1],flow),labels[1:]+int(labels[0].max())+1)
    cc[1:]=torch.where(reliable,nearest_warp(codes[:1],flow),codes[1:])
    return t,ll,cc,reliable.float().mean()


class RealRegionTextureTraining(RegionTailTraining):
    def batch_style(self,native):
        x,target,specs,flow,valid=super().batch_style(native)
        self.motion_aligned_fraction=x.new_zeros(())
        if flow is not None:
            target,self.labels,self.class_labels,fraction=align_pair(x,target,self.labels,self.class_labels,flow,valid)
            self.motion_aligned_fraction=fraction
            for spec in specs:spec['training_target_source_flow_aligned_fraction']=float(fraction)
        elif self.step%4 in (1,2):
            x=texture_intervention(x,self.labels,rng=self.rng)
            for spec in specs:spec['real_region_contrast_intervention']=[0,1.6]
        return x,target,specs,flow,valid

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        values['motion_aligned_target_fraction']=self.motion_aligned_fraction
        return loss,values

    def input_stability_loss(self,x,pred,native):
        loss,values=super().input_stability_loss(x,pred,native)
        if not native or self.step%8!=3:return loss,values
        perturbed=texture_intervention(x[:1],self.labels[:1],rng=self.rng)
        other=self.net(perturbed)[0]
        # Symmetric derivatives: the original source is also pushed to stop
        # following within-region texture. Its normal target loss remains.
        stability=F.smooth_l1_loss(other,pred[:1],beta=4/128)
        values['real_texture_output_invariance']=stability
        return loss+2*stability,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(real_region_texture_interventions=True,source_motion_aligned_pair_targets=True,
            external_runtime_filter=False,extra_inference_operations=0,visual_acceptance=False)
        torch.save(state,path)


def regression():
    labels=torch.zeros(1,64,64,dtype=torch.long);labels[:,:,32:]=1
    base=torch.where(labels[:,None]==0,-.5,.5).expand(-1,3,-1,-1).clone()
    pattern=torch.arange(64)[None,None,None,:]%2*.2-.1
    x=base+pattern
    assert (texture_intervention(x,labels,0)-base).abs().max()<2e-6
    assert torch.equal(texture_intervention(x,labels,1),x)
    assert (texture_intervention(x,labels,.4).mean((-2,-1))-x.mean((-2,-1))).abs().max()<2e-6
    flow=torch.zeros(1,2,64,64);valid=torch.ones(1,1,64,64)
    assert torch.equal(nearest_warp(labels,flow),labels)
    targets=torch.cat((base,base+.1));labs=torch.cat((labels,labels));codes=labs*81
    t,ll,cc,frac=align_pair(torch.cat((x,x)),targets,labs,codes,flow,valid)
    assert torch.equal(t[1,...,2:-2,2:-2],t[0,...,2:-2,2:-2])
    assert torch.equal(ll[1,2:-2,2:-2],ll[0,2:-2,2:-2])
    assert torch.equal(cc[1,2:-2,2:-2],cc[0,2:-2,2:-2])
    assert 0<frac<1
    print('REAL_REGION_TEXTURE_REGRESSION_PASS',flush=True)


if __name__=='__main__':
    import sys
    regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=RealRegionTextureTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
