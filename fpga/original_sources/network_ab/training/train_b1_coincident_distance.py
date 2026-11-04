"""Keep colour predicates on one contour when the target says they coincide.

Training-only within-group agreement preserves the group's predicted mean.
It is zero for a shared displaced sharp edge and cannot force different
target boundaries together. The fixed B1 inference and weight pack stay the
same. Source learning and actual-byte videos are still required for acceptance.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_reference_profile_distance import ReferenceProfileDistanceTraining,evaluate as base_evaluate,driver,preview
from .probe_b1_ordered_distance_geometry import distance_fields
from .build_b1_thirty_binary_regions import GROUPS30
from .audit_b1_coincident_ordered_edges import coincidence
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .train_b1_ternary_regions import region_codes
from .train_b1_ordered_distance_rims import distance_objective
from . import train_b1_minimum_area as m
import numpy as np


def coincident_loss(field,truth):
    terms=[];active=[]
    for (lo,hi),same in zip(GROUPS30,coincidence(truth)):
        ids=F.pad((~same).long().cumsum(1),(0,0,0,0,1,0))
        p=field[:,lo:hi]/127
        sums=torch.zeros_like(p).scatter_add(1,ids,p)
        count=torch.zeros_like(p).scatter_add(1,ids,torch.ones_like(p)).clamp_min(1)
        mean=(sums/count).gather(1,ids);mask=count.gather(1,ids)>=2
        # Backpropagate THROUGH mean: group gradients sum to zero. This term
        # changes relative contour positions, not the group's average colour.
        t=truth[:,lo:hi]/127
        target_mean=(torch.zeros_like(t).scatter_add(1,ids,t)/count).gather(1,ids)
        # Keep the small target residual admitted by the .25 tolerance; an
        # exact target must remain a zero-loss fixed point.
        error=F.smooth_l1_loss(p-mean,t-target_mean,beta=.05,reduction='none')
        terms.append((error*mask).sum());active.append(mask.sum())
    return torch.stack(terms).sum()/torch.stack(active).sum().clamp_min(1)


class CoincidentDistanceTraining(ReferenceProfileDistanceTraining):
    def additional_loss(self,p,t,native):
        # Parent computes RGB/field/profile/region terms before other forwards.
        loss,v=super().additional_loss(p,t,native)
        truth=self.supervision_distance_truth;field=self.nodes['up1_low_dw']
        if not native:truth=truth[...,10:-10,10:-10];field=field[...,10:-10,10:-10]
        shared=coincident_loss(field,truth);v['coincident_contour_disagreement']=shared
        # Seven TRAIN-crop field-gradient measurements gave weight8 only
        # .15--.27 of the existing field loss norm. Weight32 makes the new
        # contour signal comparable (.61--1.07), without changing group DC.
        return loss+32*shared,v

    def preflight_extras(self):
        result=super().preflight_extras()
        assert 'coincident_contour_disagreement' in result['values']
        return result

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(coincident_boundary_supervision=True,coincident_loss_weight=32,
            predicted_group_mean_preserved_by_loss=True,target_grouping_training_only=True,
            added_inference_operations=0,external_filter=False,visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows);device=next(net.parameters()).device
    z=F.pad(code_bits(torch.arange(1215,device=device)[:,None,None])*127,(0,0,0,0,0,2))
    pk=tuple(map(tuple,(fragment(net,z,'up1_low_dw')['rgb_head'][:,:,0,0].cpu()+128).tolist()))
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];field=net(x,all_nodes=True)['up1_low_dw']
        codes=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None].to(device)
        truth=distance_fields(codes,8);full,band=distance_objective(field,truth)
        record.update(coincident_contour_disagreement=float(coincident_loss(field,truth)),
            ordered_distance_full=float(full),ordered_distance_band=float(band))
    for key in ('coincident_contour_disagreement','ordered_distance_full','ordered_distance_band'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


def regression():
    truth=torch.zeros(1,30,24,32);ramp=(63.5+torch.arange(-16,16)*20).clamp(0,127)
    truth[:,:14]=ramp[None,None,None];truth[:,14:22]=127
    assert coincident_loss(truth,truth)==0
    common=truth.clone();common[:,:14]=torch.roll(common[:,:14],2,-1)
    assert coincident_loss(common,truth).abs()<1e-10
    split=truth.clone()
    for i in range(14):split[:,i]=torch.roll(split[:,i],i//3-2,-1)
    split.requires_grad_();loss=coincident_loss(split,truth);assert loss>.02
    loss.backward();assert torch.isfinite(split.grad).all() and split.grad.abs().sum()>0
    assert split.grad[:,:14].sum(1).abs().max()<1e-7
    # Nested but genuinely different boundaries must not be collapsed merely
    # because their luminance levels are adjacent.
    separate=torch.stack([(ramp.roll(i*3)-i).clamp(0,127) for i in range(14)])[None,:,None].expand(1,14,24,32)
    other=truth.clone();other[:,:14]=separate
    assert coincident_loss(other,other).abs()<1e-7
    assert coincident_loss(torch.zeros_like(truth),torch.zeros_like(truth))==0
    print('COINCIDENT_DISTANCE_PASS exact0/common_shift0/split_detected/group_gradient_sum0/distinct_preserved/empty0',float(loss.detach()),flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=CoincidentDistanceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
