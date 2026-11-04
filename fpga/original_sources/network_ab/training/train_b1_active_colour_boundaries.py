"""Move actual erroneous colour boundaries towards labelled boundaries.

Training-only adaptation of the directional principle in Wang et al., Active
Boundary Loss, https://arxiv.org/abs/2102.02696 . This implementation uses 15
Bernoulli colour bits and actual integer boundaries, not semantic softmax.
No dependencies, inference operators or hardware changes are introduced.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_decisive_regions import DecisiveRegionTraining
from .minimum_area_blocks import boundary_mask
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver

OFFSETS=((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))


def active_colour_boundary_loss(logits,actual_codes,target_codes):
    with torch.no_grad():
        maps=[];masks=[]
        for actual,target in zip(actual_codes.cpu().numpy(),target_codes.cpu().numpy()):
            edge=boundary_mask(target)
            if not edge.any():
                maps.append(np.zeros_like(edge,dtype=np.float32));masks.append(edge);continue
            maps.append(cv2.distanceTransform(1-edge,cv2.DIST_L2,5))
            masks.append(cv2.dilate(boundary_mask(actual),np.ones((3,3),np.uint8)))
        distance=torch.tensor(np.stack(maps),device=logits.device)
        mask=torch.tensor(np.stack(masks),device=logits.device,dtype=torch.bool)&(distance>1)
        mask[:,:2,:]=False;mask[:,-2:,:]=False;mask[:,:,:2]=False;mask[:,:,-2:]=False
        # At most 2% of crop pixels per image. Choose the farthest misplaced
        # actual boundaries so supervision is not confined to the target edge.
        for n in range(len(mask)):
            index=mask[n].flatten().nonzero().flatten();maximum=max(1,mask[n].numel()//50)
            if len(index)>maximum:
                keep=index[distance[n].flatten()[index].topk(maximum).indices]
                mask[n].zero_();mask[n].view(-1)[keep]=True
        n,y,x=mask.nonzero(as_tuple=True)
        if len(n)==0:return logits.sum()*0,dict(active_boundary_pixels=logits.new_tensor(0.))
        near=torch.stack([distance[n,y+dy,x+dx] for dy,dx in OFFSETS],1)
        best,desired=near.min(1);use=best<distance[n,y,x]-.01
        n,y,x,desired=n[use],y[use],x[use],desired[use]
        if len(n)==0:return logits.sum()*0,dict(active_boundary_pixels=logits.new_tensor(0.))
        weight=(distance[n,y,x]/10).clamp(max=1)
    center=logits.permute(0,2,3,1)[n,y,x]
    kl=[]
    for dy,dx in OFFSETS:
        other=logits.permute(0,2,3,1)[n,y+dy,x+dx].detach()
        q=other.sigmoid()
        value=q*(F.logsigmoid(other)-F.logsigmoid(center))+(1-q)*(F.logsigmoid(-other)-F.logsigmoid(-center))
        kl.append(value.sum(1).clamp_min(0))
    direction=torch.stack(kl,1)
    loss=(F.cross_entropy(direction,desired,reduction='none',label_smoothing=.1)*weight).mean()
    return loss,dict(active_boundary_direction=loss,active_boundary_pixels=logits.new_tensor(len(n)),
        active_boundary_mean_distance360=(weight*10).mean())


class ActiveColourBoundaryTraining(DecisiveRegionTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        if native:return loss,values
        logits=(self.nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
        hard=self.nodes['up2_low_dw'][:,:15,20:-20,20:-20]
        actual=torch.stack([self.predicted_codes(v-63.5) for v in hard])
        truth=self.class_labels[:,::2,::2][:,20:-20,20:-20]
        direction,details=active_colour_boundary_loss(logits,actual,truth)
        values.update(details)
        return loss+direction,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(active_colour_boundary_training=True,active_boundary_weight=1.,
            active_boundary_kind='15 Bernoulli bits; actual region boundaries; detached neighbours; training only')
        torch.save(s,path)


def regression():
    x=torch.arange(64).float()[None,None,None,:].expand(1,15,48,-1)
    wanted=(x[:,0]>=32).long();shift=torch.tensor(6.,requires_grad=True)
    logits=(x-31.5-shift)/1.5;actual=(logits[:,0].detach()>0).long()
    loss,metrics=active_colour_boundary_loss(logits,actual,wanted);loss.backward()
    assert float(shift.grad)>0,'Gradient descent must move the right-shifted boundary left'
    aligned=(x-31.5)/1.5
    exact,_=active_colour_boundary_loss(aligned,wanted,wanted)
    assert float(exact.detach())==0 and float(loss.detach())>0
    constant=torch.zeros_like(wanted)
    empty,_=active_colour_boundary_loss(logits,constant,constant)
    assert float(empty.detach())==0
    left=torch.tensor(-6.,requires_grad=True)
    left_logits=(x-31.5-left)/1.5
    left_loss,_=active_colour_boundary_loss(left_logits,(left_logits[:,0].detach()>0).long(),wanted)
    left_loss.backward();assert float(left.grad)<0
    print('ACTIVE_COLOUR_BOUNDARY_REGRESSION',dict(loss=float(loss.detach()),
        shift_gradient=float(shift.grad),left_shift_gradient=float(left.grad),
        aligned=float(exact.detach()),constant=float(empty.detach())),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_decisive_regions.py','train_b1_native_distance_boundaries.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py','train_b1_ordered_regions.py',
        'build_b1_ordered_palette.py','minimum_area_blocks.py','train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=ActiveColourBoundaryTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
