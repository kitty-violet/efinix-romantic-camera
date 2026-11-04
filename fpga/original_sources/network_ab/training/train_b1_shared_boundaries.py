"""Align colour-level crossings which describe the SAME target contour.

Custom training-only constraint, not an implementation of Active Boundary Loss.
It was motivated by checking boundary alignment as well as region accuracy:
https://arxiv.org/abs/2102.02696 . No new inference nodes or postprocessing.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_decisive_regions import DecisiveRegionTraining
from .train_b1_ordered_regions import evaluate,preview
from .build_b1_ordered_palette import bits
from . import train_b1_late_region_supervision as driver


def shared_boundary_loss(logits,truth):
    """Only couple bit pairs that are identical/complementary near a contour.

    Binary colour levels on opposite sides of the same target contour should
    cross at the same location. Different target contours are never coupled.
    RGB/classification losses still determine colour and boundary position.
    """
    indices=torch.triu_indices(logits.shape[1],logits.shape[1],1,device=logits.device)
    i,j=indices;prob=logits.sigmoid()
    with torch.no_grad():
        hard=(truth>=.5).float()
        xor=(hard[:,i]-hard[:,j]).abs()
        relation=F.avg_pool2d(xor,5,1,2)
        hi=F.max_pool2d(hard,5,1,2)
        lo=-F.max_pool2d(-hard,5,1,2)
        mixed=(hi-lo)>.5
        same=relation<1e-5;opposite=relation>1-1e-5
        mask=((same|opposite)&mixed[:,i]&mixed[:,j]).float()
        # Do not let zero-padding invent an outside-neighbour relationship.
        mask[...,:2,:]=0;mask[...,-2:,:]=0;mask[...,:,:2]=0;mask[...,:,-2:]=0
    other=torch.where(opposite,1-prob[:,j],prob[:,j])
    error=(prob[:,i]-other).abs()
    loss=(error*mask).sum()/mask.sum().clamp_min(1)
    return loss,dict(shared_boundary_disagreement=loss,
        shared_boundary_pair_coverage=mask.mean())


class SharedBoundaryTraining(DecisiveRegionTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        if native:return loss,values
        logits=(self.nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
        wanted=F.avg_pool2d(self.bit_table[self.class_labels].permute(0,3,1,2),2)[...,20:-20,20:-20]
        shared,details=shared_boundary_loss(logits,wanted)
        values.update(details)
        return loss+2*shared,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(shared_contour_bit_alignment=True,shared_contour_weight=2.,
            shared_contour_scope='5x5 identical or complementary target bit boundaries; training only')
        torch.save(s,path)


def regression():
    x=torch.arange(64).float()[None,None,None,:].expand(1,15,48,-1)
    truth=(x>=32).float();truth[:,7:]=1-truth[:,7:]
    signs=torch.ones(1,15,1,1);signs[:,7:]=-1
    good=(x-31.5)*signs
    delayed=good.clone();delayed[:,:3]=(x[:,:3]-35.5)
    narrow=good.clone();narrow[:,0]=(x[:,0]-20.5)
    values={k:float(shared_boundary_loss(v,truth)[0]) for k,v in
        [('same_contour',good),('extra_colour_band',delayed)]}
    assert values['same_contour']<1e-6 and values['extra_colour_band']>.02
    constant=torch.zeros_like(truth)
    assert float(shared_boundary_loss(good,constant)[0])==0
    displacement=torch.tensor(3.,requires_grad=True)
    ramp=good.clone();ramp[:,:3]=x[:,:3]-31.5-displacement
    cost,_=shared_boundary_loss(ramp,truth);cost.backward()
    assert float(displacement.grad)>0,'Descent must pull the delayed boundary back'
    # An unrelated single-bit contour should not be forced onto another one.
    two=truth.clone();two[:,0]=(x[:,0]>=20).float()
    isolated=(x-31.5)*signs;isolated[:,0]=x[:,0]-19.5
    assert float(shared_boundary_loss(isolated,two)[0])<1e-6
    print('SHARED_BOUNDARY_REGRESSION',values,'delay_gradient',float(displacement.grad),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_decisive_regions.py','train_b1_native_distance_boundaries.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py','train_b1_ordered_regions.py',
        'build_b1_ordered_palette.py','minimum_area_blocks.py','train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=SharedBoundaryTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
