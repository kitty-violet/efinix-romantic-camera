"""Make actual connected colour cleanup the primary learning objective.

No pixelwise teacher classification/distance regression: the teacher anchors
region-average colours and their contrast. Detached cleanup of the current
integer prediction defines a simpler colour partition. It is training-only;
the delivered B1 performs one ordinary pass with no external postprocessing.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_native_distance_boundaries import NativeDistanceTraining
from .train_b1_decisive_regions import modal_region_cleanup
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def colour_anchor(p,t,labels):
    losses=[];contrasts=[]
    for pred,target,lab in zip(p,t,labels):
        _,idx=torch.unique(lab.flatten(),return_inverse=True);n=int(idx.max())+1
        count=torch.bincount(idx,minlength=n).float();index=idx[None].expand(3,-1)
        pm=pred.new_zeros(3,n).scatter_add(1,index,pred.flatten(1))/count[None]
        tm=target.new_zeros(3,n).scatter_add(1,index,target.flatten(1))/count[None]
        weight=count.sqrt();err=(pm-tm).abs().mean(0)
        losses.append((err*weight).sum()/weight.sum())
        grid=idx.reshape(lab.shape);pairs=[]
        for a,b in ((grid[:,1:],grid[:,:-1]),(grid[1:],grid[:-1])):
            mask=a!=b;u,v=a[mask],b[mask]
            pairs.append(torch.minimum(u,v)*n+torch.maximum(u,v))
        pairs=torch.unique(torch.cat(pairs));a=pairs//n;b=pairs%n
        if len(pairs):
            expected=tm[:,a]-tm[:,b];actual=pm[:,a]-pm[:,b]
            importance=expected.abs().mean(0).clamp_min(.02)
            contrasts.append(((actual-expected).abs().mean(0)*importance).sum()/importance.sum())
        else:contrasts.append(pred.sum()*0)
    return torch.stack(losses).mean(),torch.stack(contrasts).mean()


def coarse_appearance(p,t,mode):
    value=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(t,16))
    return .25*value,dict(coarse_layout_anchor=value)


class ProjectedRegionTraining(NativeDistanceTraining):
    appearance_loss=staticmethod(coarse_appearance)

    def additional_loss(self,p,target,native):
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        hard=self.nodes['up2_low_dw'][:,:15]
        palette=(self.palette.detach().cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
        classes=[];changed=[];fractions=[]
        with torch.no_grad():
            for item in hard:
                code=self.predicted_codes(item-63.5).cpu().numpy()
                clean,_=modal_region_cleanup(code,palette,64,2)
                change=clean!=code
                if not native:clean=clean[20:-20,20:-20];change=change[20:-20,20:-20]
                classes.append(torch.tensor(clean,device=p.device));changed.append(torch.tensor(change,device=p.device))
                fractions.append(float(change.mean()))
            desired=self.bit_table[torch.stack(classes)].permute(0,3,1,2)
            change=torch.stack(changed)[:,None]
            weights=1+3*change.float()
        labels=self.labels
        if not native:
            logits=logits[...,20:-20,20:-20];labels=labels[...,40:-40,40:-40]
        ce=F.binary_cross_entropy_with_logits(logits,desired,reduction='none')
        cleanup=(ce*weights).sum()/(15*weights.sum())
        anchor,contrast=colour_anchor(p,target,labels)
        return 5*cleanup+2*anchor+contrast,dict(primary_region_projection=cleanup,
            regional_colour_anchor=anchor,regional_contrast_anchor=contrast,
            projected_changed_fraction=p.new_tensor(sum(fractions)/len(fractions)))

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(primary_region_projection=True,teacher_pixel_classification=False,
            teacher_distance_regression=False,objective='self modal projection5 + region colour2 + adjacent colour contrast1 + coarse16 RGB.25',
            minimum_area_inference_guarantee=False)
        torch.save(s,path)


def regression():
    labels=torch.zeros(1,64,64,dtype=torch.long);labels[:,:,32:]=1
    target=torch.full((1,3,64,64),-.5);target[:,:,:,32:]=.5
    a,b=colour_anchor(target,target,labels);c,d=colour_anchor(target*0,target,labels)
    assert float(a+b)==0 and float(c)>.4 and float(d)>.9
    # A spurious island must receive a gradient towards its surrounding code.
    code=np.zeros((24,24),np.int64);code[10:13,10:13]=25
    palette=np.tile(np.arange(200,dtype=np.uint8)[:,None],(1,3))
    clean,_=modal_region_cleanup(code,palette,64,2);assert not clean.any()
    logits=torch.full((1,1,24,24),-2.);logits[:,:,10:13,10:13]=2;logits.requires_grad_(True)
    desired=torch.tensor(clean[None,None]>0).float()
    F.binary_cross_entropy_with_logits(logits,desired).backward()
    assert float(logits.grad[0,0,11,11])>0
    print('PRIMARY_PROJECTION_REGRESSION',dict(correct=0,constant_colour_error=float(c),
        constant_contrast_error=float(d),island_removed=True,gradient_correct=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_native_distance_boundaries.py','train_b1_decisive_regions.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py',
        'train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py',
        'train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=ProjectedRegionTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
