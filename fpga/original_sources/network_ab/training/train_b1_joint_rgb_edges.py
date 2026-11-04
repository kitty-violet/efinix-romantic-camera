"""Match RGB edge vectors at one shared location, not per-channel maxima.

The old tolerant edge term can accept red/green/blue transitions at different
nearby pixels, which admits coloured fringes. Keep a small positional tolerance
but require all three components to come from the same candidate position.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_ternary_region_balance import TernaryRegionBalanceTraining
from .train_b1_joint_region_colour import evaluate,preview
from .train_b1_region_formation import formation_loss
from . import train_b1_late_region_supervision as driver


def rgb_edge_vector_loss(p,t,radius=4):
    losses=[]
    for axis in (-1,-2):
        gp,gt=torch.diff(p,dim=axis),torch.diff(t,dim=axis)
        mask=(gt.abs().amax(1)>6/128).float()
        padded=F.pad(gp,(radius,radius,0,0) if axis==-1 else (0,0,radius,radius),mode='replicate')
        options=[]
        for offset in range(2*radius+1):
            candidate=padded.narrow(axis,offset,gp.shape[axis])
            options.append((candidate-gt).abs().mean(1))
        # One index for the whole RGB vector. No independent channel offsets.
        distance=torch.stack(options).amin(0)
        losses.append((distance*mask).sum()/mask.sum().clamp_min(1))
    return torch.stack(losses).mean()


class JointRGBEdgeTraining(TernaryRegionBalanceTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        vector=rgb_edge_vector_loss(p,target)
        values['joint_rgb_edge_vector']=vector
        return loss+2*vector,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(joint_rgb_boundary_location=True,vector_edge_radius=4,visual_acceptance=False)
        torch.save(s,path)


def regression():
    t=torch.zeros(1,3,64,96);t[:,:,:,48:]=torch.tensor([.6,.4,.2])[None,:,None,None]
    labels=torch.zeros(1,64,96,dtype=torch.long);labels[:,:,48:]=1
    split=t.clone();split[:,0]=t[:,0].roll(-3,-1);split[:,2]=t[:,2].roll(3,-1)
    # Do not include roll wrap-around in the comparison crop.
    shifted=t.roll(3,-1)
    blur=F.avg_pool2d(F.pad(t,(4,4,4,4),mode='replicate'),9,1)
    cases=dict(exact=t,whole_edge_shift3=shifted,channel_split=split,blur9=blur)
    values={}
    for key,p in cases.items():
        p=p[...,12:-12];tt=t[...,12:-12];ll=labels[...,12:-12]
        _,old=formation_loss(p,tt,ll)
        values[key]=dict(old_independent=float(old['missing_sharp_boundary']),joint_vector=float(rgb_edge_vector_loss(p,tt)))
    assert values['exact']['joint_vector']==0 and values['whole_edge_shift3']['joint_vector']==0
    assert values['channel_split']['old_independent']==0 and values['channel_split']['joint_vector']>.02
    assert values['blur9']['joint_vector']>.02
    p=split[...,12:-12].clone().requires_grad_();loss=rgb_edge_vector_loss(p,t[...,12:-12]);loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    print('JOINT_RGB_EDGE_REGRESSION_PASS',values,flush=True)
    return values


if __name__=='__main__':
    import sys
    if '--check' in sys.argv:regression()
    else:
        regression()
        driver.main(training_factory=JointRGBEdgeTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
