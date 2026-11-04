"""Audit-labelled region tails: spend gradients on incomplete colour blocks.

Whole-image pixel averages can improve while islands inside each region remain.
The additional objective gives each sufficiently large target region equal
weight, using its worst quarter of pure-footprint count errors. Every selected
pixel retains its audited target class; no self-generated colour is taught.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_native_coverage_counts import NativeCoverageCountTraining,schedule_regression
from .train_b1_truth_area_counts import regression as area_regression
from .train_b1_count_regions import count_targets,log_count_probability,regression as count_regression
from .build_b1_ordered_palette import GROUPS
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def region_tail_loss(logits,targets,full_regions,native):
    nll=torch.stack([-(t*log_count_probability(logits[:,a:z])).sum(1)
        for (a,z),t in zip(GROUPS,targets)]).mean(0)
    labels=full_regions[:,None].float()
    hi=F.max_pool2d(labels,2);lo=-F.max_pool2d(-labels,2)
    regions=lo[:,0].long();pure=(lo==hi)[:,0]
    if not native:regions=regions[...,20:-20,20:-20];pure=pure[...,20:-20,20:-20]
    assert regions.shape==nll.shape
    losses=[];used=[]
    for values,ids,valid in zip(nll,regions,pure):
        values=values[valid];ids=ids[valid]
        if not len(ids):losses.append(nll.sum()*0);used.append(0);continue
        _,ids=torch.unique(ids,return_inverse=True,sorted=True)
        counts=torch.bincount(ids);active=counts>=32
        # Stable lexicographic (region ascending, detached loss descending).
        order=torch.argsort(values.detach(),descending=True,stable=True)
        order=order[torch.argsort(ids[order],stable=True)]
        starts=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        ranks=torch.arange(len(ids),device=ids.device)-torch.repeat_interleave(starts,counts)
        take=active[ids[order]]&(ranks<((counts[ids[order]]+3)//4))
        selected=order[take];selected_ids=ids[selected]
        totals=values.new_zeros(len(counts)).scatter_add(0,selected_ids,values[selected])
        number=torch.bincount(selected_ids,minlength=len(counts)).clamp_min(1)
        losses.append((totals/number)[active].mean() if active.any() else nll.sum()*0)
        used.append(int(active.sum()))
    return torch.stack(losses).mean(),dict(supervised_tail_regions=sum(used))


class RegionTailCountTraining(NativeCoverageCountTraining):
    def additional_loss(self,p,target,native):
        base,values=super().additional_loss(p,target,native)
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        truth=count_targets(self.class_labels)
        if not native:
            logits=logits[...,20:-20,20:-20];truth=[v[...,20:-20,20:-20] for v in truth]
        tail,details=region_tail_loss(logits,truth,self.labels,native)
        values.update(region_tail_count_ce=tail,
            region_tail_active_regions=tail.new_tensor(details['supervised_tail_regions']))
        return base+.35*tail,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(region_tail_count_supervision=True,region_tail_fraction=.25,region_tail_min_core360=32,
            region_tail_weight=.35,objective='existing count/stability plus .35 equal-region worst-quarter truth count CE')
        torch.save(state,path)


def regression():
    codes=torch.zeros(1,32,32,dtype=torch.long);codes[:,:,16:]=199
    regions=(codes>0).long();targets=count_targets(codes)
    good=torch.where(codes[:,None,::2,::2]>0,6.,-6.).expand(-1,15,-1,-1).clone()
    bad=good.clone();bad[:,:,4:8,4:8]=6.;bad.requires_grad_(True)
    a,_=region_tail_loss(good,targets,regions,True);b,stats=region_tail_loss(bad,targets,regions,True)
    assert b>a+.2 and stats['supervised_tail_regions']==2
    b.backward();assert (bad.grad[:,:,4:8,4:8]>0).all()
    renamed=regions*37+9;c,_=region_tail_loss(bad.detach(),targets,renamed,True)
    assert torch.equal(b.detach(),c)
    # A constant all-one-region image has a finite differentiable objective.
    d,_=region_tail_loss(torch.zeros_like(good,requires_grad=True),targets,regions*0,True)
    assert torch.isfinite(d);d.backward()
    print('REGION_TAIL_REGRESSION target_direction / region_renumber / finite PASS',flush=True)


if __name__=='__main__':
    regression();area_regression();count_regression();schedule_regression();here=Path(__file__).parent
    names=['train_b1_native_coverage_counts.py','train_b1_truth_area_counts.py','train_b1_temporal_count_regions.py',
        'train_b1_count_regions.py','prepare_b1_temporal_classes.py','train_b1_native_distance_boundaries.py',
        'train_b1_decisive_regions.py','probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py','train_b1_ordered_regions.py',
        'build_b1_ordered_palette.py','minimum_area_blocks.py','train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=RegionTailCountTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
