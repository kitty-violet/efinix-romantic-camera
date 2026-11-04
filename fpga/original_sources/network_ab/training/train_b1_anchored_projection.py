"""Correct only spurious regions; anchor colour before saturating decisions.

Self-label confidence everywhere has a trivial colour-collapse equilibrium.
Here unchanged predictions receive NO self-label reinforcement. Ground-truth
regions anchor their mean decision fields directly, so a confident wrong bit
still receives a corrective derivative. This is a training objective only;
the fixed ordinary B1 byte graph and all deployed operations are unchanged.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_projected_regions import ProjectedRegionTraining,colour_anchor
from .train_b1_decisive_regions import modal_region_cleanup
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def region_decision_anchor(logits,truth,labels,valid=None):
    values=[]
    if valid is None:valid=torch.ones_like(labels,dtype=torch.bool)
    for field,wanted,lab,mask in zip(logits,truth,labels,valid):
        keep=mask.flatten()
        if not keep.any():
            values.append(field.sum()*0);continue
        _,ids=torch.unique(lab.flatten()[keep],return_inverse=True);n=int(ids.max())+1
        count=torch.bincount(ids,minlength=n).float();index=ids[None].expand(15,-1)
        mean=field.new_zeros(15,n).scatter_add(1,index,field.flatten(1)[:,keep])/count[None]
        target=field.new_zeros(15,n).scatter_add(1,index,wanted.flatten(1)[:,keep])/count[None]
        weight=count.sqrt()
        ce=F.binary_cross_entropy_with_logits(mean,target,reduction='none').mean(0)
        values.append((ce*weight).sum()/weight.sum())
    return torch.stack(values).mean()


def changed_only_ce(logits,desired,change):
    error=F.binary_cross_entropy_with_logits(logits,desired,reduction='none')
    return (error*change).sum()/(logits.shape[1]*change.sum()).clamp_min(1)


class AnchoredProjectionTraining(ProjectedRegionTraining):
    def additional_loss(self,p,target,native):
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        hard=self.nodes['up2_low_dw'][:,:15]
        palette=(self.palette.detach().cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
        classes=[];changes=[]
        with torch.no_grad():
            for item in hard:
                code=self.predicted_codes(item-63.5).cpu().numpy()
                clean,_=modal_region_cleanup(code,palette,64,2)
                change=clean!=code
                if not native:clean=clean[20:-20,20:-20];change=change[20:-20,20:-20]
                classes.append(torch.tensor(clean,device=p.device));changes.append(torch.tensor(change,device=p.device))
            desired=self.bit_table[torch.stack(classes)].permute(0,3,1,2)
            change=torch.stack(changes)[:,None].float()
            truth=F.avg_pool2d(self.bit_table[self.class_labels].permute(0,3,1,2),2)
            # A 2x2 footprint touching different teacher regions cannot provide
            # one unambiguous region index; leave it out of region averaging.
            labels=self.labels[:,::2,::2]
            lf=self.labels[:,None].float()
            valid=(F.max_pool2d(lf,2)==-F.max_pool2d(-lf,2))[:,0]
        label720=self.labels
        if not native:
            logits=logits[...,20:-20,20:-20];truth=truth[...,20:-20,20:-20]
            labels=labels[...,20:-20,20:-20];label720=label720[...,40:-40,40:-40]
            valid=valid[...,20:-20,20:-20]
        correction=changed_only_ce(logits,desired,change)
        decision=region_decision_anchor(logits,truth,labels,valid)
        colour,contrast=colour_anchor(p,target,label720)
        return 2*correction+2*decision+2*colour+contrast,dict(
            changed_only_projection=correction,decision_colour_anchor=decision,
            regional_colour_anchor=colour,regional_contrast_anchor=contrast,
            projected_changed_fraction=change.mean())

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(anchored_region_projection=True,self_label_confidence_on_unchanged_pixels=False,
            predecision_region_colour_anchor=True,minimum_area_inference_guarantee=False,
            objective='changed-only modal CE2 + region-mean-logit CE2 + region RGB2 + adjacent contrast1 + coarse16 RGB.25')
        torch.save(s,path)


def regression():
    logits=torch.full((1,15,16,16),10.,requires_grad=True)
    truth=torch.zeros_like(logits);labels=torch.zeros(1,16,16,dtype=torch.long)
    stable=region_decision_anchor(logits,truth,labels)
    grad=torch.autograd.grad(stable,logits,retain_graph=True)[0].sum()
    old=logits.sigmoid().mean();old_grad=torch.autograd.grad(old,logits)[0].sum()
    assert float(grad)>.99 and float(old_grad)<.0001
    x=torch.full((1,15,16,16),2.,requires_grad=True)
    nochange=changed_only_ce(x,torch.ones_like(x),torch.zeros(1,1,16,16))
    nochange.backward();assert not x.grad.any() and float(nochange)==0
    x.grad.zero_();mask=torch.zeros(1,1,16,16);mask[:,:,7:9,7:9]=1
    correction=changed_only_ce(x,torch.zeros_like(x),mask);correction.backward()
    assert x.grad[:,:,7:9,7:9].min()>0 and not x.grad[:,:,:7].any()
    exact=region_decision_anchor(-logits.detach(),truth,labels)
    assert float(exact)<.0001
    print('ANCHORED_PROJECTION_REGRESSION',dict(wrong_confident_direct_gradient=float(grad),
        saturated_rgb_probability_gradient=float(old_grad),unchanged_self_gradient=0,
        correction_sign_correct=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_projected_regions.py','train_b1_native_distance_boundaries.py','train_b1_decisive_regions.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py',
        'train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py',
        'train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=AnchoredProjectionTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
