"""Supervise mixed boundary cells by ordinal area, not averaged colour code.

At180p one cell can contain several720p regions. Averaging their colour indices
before a nonlinear hinge does not equal the area of each region. Keep the
shared-coordinate target only for unambiguous cells; supervise actual post-vote
ordered predicates with the native target's4x4 occupancy everywhere.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_ordinal_vote_hinges import OrdinalVoteTraining,driver,evaluate,preview
from .build_b1_thirty_binary_regions import code_bits,GROUPS30

def occupancy_loss(pred,code):
    target=F.avg_pool2d(code_bits(code),4)
    return F.smooth_l1_loss(pred[:,:30]/127,target,beta=.1)

def homogeneous_coordinate_loss(hinge,code):
    coordinate=torch.stack([hinge[:,a:b].sum(1)/127 for a,b in GROUPS30],1)
    target=F.avg_pool2d(torch.stack((code//81,code//9%9,code%9),1).float(),4)
    low=-F.max_pool2d(-code[:,None].float(),4,4);high=F.max_pool2d(code[:,None].float(),4,4)
    valid=(low==high).float();err=F.smooth_l1_loss(coordinate,target,beta=.5,reduction='none')
    return (err*valid).sum()/(3*valid.sum()).clamp_min(1)

class BoundaryOccupancyTraining(OrdinalVoteTraining):
    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        code=self.class_labels;before=self.nodes['res3_conv1.pw'];after=self.nodes['up1_low_dw']
        if not native:code=code[...,40:-40,40:-40];before=before[...,10:-10,10:-10];after=after[...,10:-10,10:-10]
        solid=homogeneous_coordinate_loss(before,code);area=occupancy_loss(after,code)
        loss=loss-3*v['shared_coordinate_colour']+3*solid+6*area
        v.update(homogeneous_coordinate_colour=solid,post_vote_ordinal_occupancy=area)
        return loss,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(boundary_occupancy_training=True,coordinate_target_scope='homogeneous4x4cells',
          post_vote_ordinal_occupancy_weight=6,visual_acceptance=False);torch.save(s,path)

def regression():
    from .audit_b1_common_coordinate_boundaries import hinges
    code=torch.full((1,16,24),5*81+4*9+4,dtype=torch.long);code[...,9:]=9*81+4*9+4
    truth=F.avg_pool2d(code_bits(code),4);assert occupancy_loss(truth*127,code)==0
    mixed=truth[...,2];assert ((mixed>0)&(mixed<1)).any()
    coord=F.avg_pool2d(torch.stack((code//81,code//9%9,code%9),1).float(),4)
    h=hinges(coord);assert homogeneous_coordinate_loss(h,code)==0
    assert occupancy_loss(h,code)>0  # mean colour is not ordinal area at a boundary
    # Altering only the ambiguous cell must not create a false coordinate target.
    h2=h.clone();h2[...,2]=0;assert homogeneous_coordinate_loss(h2,code)==0
    leaf=torch.zeros_like(h,requires_grad=True);loss=occupancy_loss(leaf,code);loss.backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad.abs().sum()>0
    print('BOUNDARY_OCCUPANCY_REGRESSION_PASS exact_area0/mixed_cell/no_false_mean_constraint/gradient',flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:driver.main(training_factory=BoundaryOccupancyTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
