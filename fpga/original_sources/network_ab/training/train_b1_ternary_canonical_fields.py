"""Supervise each ternary region field before the sharp fixed decoder.

Reuses the earlier independent-field idea, now with 15/9/9 colour levels and
the tested three-state spatial vote. The previous count DP supervised totals
only; here canonical field states and their pre-threshold margins are explicit.
No labels, distances, or additional operators are used during inference.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_region_tail_flatness import RegionTailTraining,evaluate,preview
from .probe_b1_ternary_field_targets import canonical_states
from .train_b1_ternary_regions import log_states
from . import train_b1_late_region_supervision as driver


def field_objective(pre,states):
    #16 is the tested canonical margin, not32 (which adds excess edge bands).
    coordinate=(pre-63.5)/16+1
    hot=F.one_hot(states.long(),3).permute(0,1,4,2,3).float()
    n,c,k,h,w=hot.shape
    truth=F.avg_pool2d(hot.flatten(0,1),4).reshape(n,c,k,h//4,w//4)
    freq=truth.mean((0,3,4),keepdim=True)
    weight=freq.clamp_min(.02).rsqrt()
    weight=(weight/(freq*weight).sum(2,keepdim=True)).clamp(.25,4).detach()
    logp=log_states(coordinate,.20)
    weighted=truth*weight
    ce=-(weighted*logp).sum()/weighted.sum().clamp_min(1)
    desired=F.avg_pool2d(states,4)
    # Keep margin magnitudes bounded rather than driving all0/2 fields to
    # arbitrary extreme values that reintroduce unequal transition widths.
    margin=F.smooth_l1_loss(coordinate,desired,beta=.25)
    return ce+.3*margin,dict(canonical_field_ce=ce,canonical_field_margin=margin)


class CanonicalTernaryTraining(RegionTailTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        key='up1_low_pw';s=(getattr(self.net,key+'_m')/131072).float().flatten()
        w=self.net.round(self.net.weight[key]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
        b=self.net.round(self.net.bias[key]/s)*s
        pre=self.net.round(F.conv2d(self.nodes['up1_low_dw'],w,b))[:,:15]
        states=canonical_states(self.class_labels)
        if not native:pre=pre[...,10:-10,10:-10];states=states[...,40:-40,40:-40]
        field,terms=field_objective(pre,states)
        # Replace only the aggregate count classification part; RGB/regions,
        # common edge vectors, temporal consistency and phase checks remain.
        loss=loss-self.args.classification*values['ternary_count_ce']+self.args.classification*field
        values.update(terms)
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(canonical_ternary_fields=True,canonical_margin=16,
            aggregate_count_supervision=False,extra_inference_operations=0,visual_acceptance=False)
        torch.save(state,path)


def regression():
    for code in (0,40,608,1214):
        ids=torch.full((1,64,64),code,dtype=torch.long);states=canonical_states(ids)
        counts=torch.stack((states[:,:7].sum(1),states[:,7:11].sum(1),states[:,11:15].sum(1)),1)
        assert torch.equal(counts[:,0]*81+counts[:,1]*9+counts[:,2],ids.float())
    ids=torch.zeros(1,64,64,dtype=torch.long);ids[:,:,32:]=1214
    states=canonical_states(ids);good=(F.avg_pool2d(states,4)-1)*16+63.5
    correct=field_objective(good,states)[0]
    wrong=good.flip(-1).detach().requires_grad_();loss=field_objective(wrong,states)[0];loss.backward()
    assert loss>correct+2 and torch.isfinite(wrong.grad).all() and wrong.grad.abs().sum()>0
    assert wrong.grad[...,:8].mean()>0 and wrong.grad[...,8:].mean()<0
    print('CANONICAL_TERNARY_REGRESSION_PASS',float(correct),float(loss.detach()),flush=True)


if __name__=='__main__':
    import sys
    regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=CanonicalTernaryTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
