"""Group existing B1 features inside labelled colour regions during training.

Inspired by the pull/push principle of De Brabandere et al. (2017),
https://arxiv.org/abs/1708.02551 . This is an independent, colour-adjacency
variant, not their instance-segmentation implementation. There is no learned
auxiliary head, inference clustering or new operator in the exported B1.
"""
from pathlib import Path
import math
import torch
from torch.nn import functional as F
from .train_b1_region_sets import RegionSetTraining
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def feature_regions(feature,full_labels,colour,native):
    """Keep full4x4 region coverage; do not discard every small region's core."""
    labels=full_labels[:,None].float()
    hi=F.max_pool2d(labels,4);lo=-F.max_pool2d(-labels,4)
    valid=(hi==lo)[:,0];labels=hi[:,0].long()
    target=F.avg_pool2d(colour,4)
    if not native:
        feature=feature[...,10:-10,10:-10];labels=labels[...,10:-10,10:-10]
        valid=valid[...,10:-10,10:-10];target=target[...,10:-10,10:-10]
    pulls=[];pushes=[];tails=[];coverages=[]
    for emb,lab,mask,rgb in zip(feature,labels,valid,target):
        flat=lab.flatten();_,index=torch.unique(flat,return_inverse=True)
        n=int(index.max())+1;good=mask.flatten().float()
        count=torch.bincount(index,weights=good,minlength=n)
        active=count>0;idx=index[None].expand(emb.shape[0],-1)
        # RMS distance is normalized in activation codes, not with adaptive
        # per-image whitening that could erase actual region contrast.
        z=emb.flatten(1)/16
        mu=z.new_zeros(z.shape[0],n).scatter_add(1,idx,z*good)/count.clamp_min(1)[None]
        distance=((z-mu[:,index]).square().mean(0)+1e-8).sqrt()
        variance=F.relu(distance-.08).square()*good
        per_region=variance.new_zeros(n).scatter_add(0,index,variance)/count.clamp_min(1)
        maximum=variance.new_zeros(n).scatter_reduce(0,index,variance,reduce='amax',include_self=True)
        pull=per_region[active].mean() if active.any() else z.sum()*0
        tail=maximum[active].mean() if active.any() else z.sum()*0
        colours=rgb.flatten(1);cm=colours.new_zeros(3,n).scatter_add(1,index[None].expand(3,-1),colours*good)/count.clamp_min(1)[None]
        grid=index.reshape(lab.shape);pairs=[]
        # Labels identify touching regions even where the4x4 footprint crosses
        # the boundary. Only means with actual pure samples are eligible.
        for aa,bb in ((grid[:,1:],grid[:,:-1]),(grid[1:],grid[:-1])):
            sel=(aa!=bb)&active[aa]&active[bb]
            u,v=aa[sel],bb[sel]
            pairs.append(torch.minimum(u,v)*n+torch.maximum(u,v))
        pairs=torch.unique(torch.cat(pairs));u=pairs//n;v=pairs%n
        if len(pairs):
            apart=((mu[:,u]-mu[:,v]).square().mean(0)+1e-8).sqrt()
            colour_gap=(cm[:,u]-cm[:,v]).abs().mean(0)
            # Same-colour regions are allowed to meet. Different labelled
            # colours must retain contrast, proportional to their target gap.
            margin=(2*colour_gap).clamp(max=.75).detach()
            push=F.relu(margin-apart).square().mean()
        else:push=z.sum()*0
        pulls.append(pull);tails.append(tail);pushes.append(push);coverages.append(good.mean())
    pull=torch.stack(pulls).mean();tail=torch.stack(tails).mean();push=torch.stack(pushes).mean()
    return pull+.25*tail+push,dict(region_feature_pull=pull,region_feature_worst=tail,
        region_feature_separation=push,region_feature_coverage=torch.stack(coverages).mean())


class RegionFeatureTraining(RegionSetTraining):
    def additional_loss(self,p,target,native):
        base,values=super().additional_loss(p,target,native)
        colour=self.palette[self.class_labels].permute(0,3,1,2)
        grouping,details=feature_regions(self.nodes['up1_low_dw'],self.labels,colour,native)
        values.update(details)
        return base+2*grouping,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(region_feature_grouping=True,feature_grouping_weight=2,
            feature_grouping_training_only=True,auxiliary_inference_ops=0)
        torch.save(s,path)


def regression():
    labels=torch.zeros(1,64,64,dtype=torch.long);labels[:,:,32:]=1
    colour=torch.full((1,3,64,64),-.5);colour[:,:,:,32:]=.5
    clean=torch.zeros(1,4,16,16);clean[:,:,:,8:]=16
    noisy=clean.clone();noisy[:,:,3:6,3:6]=24
    noisy.requires_grad_(True)
    a,_=feature_regions(clean,labels,colour,True)
    b,_=feature_regions(noisy,labels,colour,True)
    c,_=feature_regions(torch.zeros_like(clean),labels,colour,True)
    assert float(a)<1e-6 and float(b.detach())>.01 and float(c)>.1
    b.backward();assert float(noisy.grad[:,:,4,4].sum())>0
    # Arbitrary label numbering cannot change a region-based objective.
    remapped=torch.where(labels==0,torch.tensor(71),torch.tensor(9))
    other,_=feature_regions(noisy.detach(),remapped,colour,True)
    assert abs(float(other)-float(b.detach()))<1e-6
    print('REGION_FEATURE_REGRESSION',dict(clean=float(a),noise=float(b.detach()),
        collapse=float(c),label_permutation_equal=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_region_sets.py','train_b1_native_distance_boundaries.py',
        'train_b1_active_colour_boundaries.py','train_b1_decisive_regions.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py',
        'train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py',
        'train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=RegionFeatureTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
