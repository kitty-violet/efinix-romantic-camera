"""Complete region supervision, including narrow regions and actual islands.

The previous field-interior erosion excluded64% of sampled positions. This
experiment supervises every field position and normalizes each region before
combining them. Connected component corrections are training-only targets;
the exported network still executes exactly the original B1 graph.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_flat_colour_fields import FlatColourTraining
from . import train_b1_ordered_regions as ordered
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .build_b1_ordered_palette import GROUPS,offsets


def complete_fields(field,truth,labels):
    labels=F.interpolate(labels[:,None].float(),size=field.shape[-2:],mode='nearest')[:,0].long()
    flats=[];colours=[]
    for p,t,lab in zip(field,truth,labels):
        ids=lab.flatten();n=int(ids.max())+1;ix=ids[None].expand(3,-1)
        count=torch.bincount(ids,minlength=n).float().clamp_min(1)
        mean=p.new_zeros(3,n).scatter_add(1,ix,p.flatten(1))/count[None]
        target=t.new_zeros(3,n).scatter_add(1,ix,t.flatten(1))/count[None]
        # Every position contributes. sqrt(area) limits dominance by walls/sky
        # without letting a one-cell region dominate a substantial object.
        weight=count[ids].rsqrt();normalizer=3*weight.sum()
        flats.append(((p.flatten(1)-mean[:,ids]).abs()*weight).sum()/normalizer)
        colours.append(((mean[:,ids]-target[:,ids]).abs()*weight).sum()/normalizer)
    point=F.smooth_l1_loss(field,truth,beta=.25)
    return dict(whole_field_flat=torch.stack(flats).mean(),whole_field_colour=torch.stack(colours).mean(),whole_field_point=point)


class CompleteRegionTraining(FlatColourTraining):
    def additional_loss(self,p,target,native):
        key='up1_low_pw';w,b=self.net.weight[key],self.net.bias[key]
        scale=(getattr(self.net,key+'_m')/131072).float().flatten()
        qw=self.net.round(w/scale[:,None,None,None]).clamp(-127,127)*scale[:,None,None,None]
        qb=self.net.round(b/scale)*scale;off=offsets(w.device)
        cw=torch.stack([qw[a:z].mean(0) for a,z in GROUPS])
        cb=torch.stack([(qb[a:z]-off[a:z]).mean() for a,z in GROUPS])
        coord=F.conv2d(self.nodes['up1_low_dw'],cw,cb)
        # Choose the region at the field sample, rather than average several
        # different colour codes into an unwanted intermediate colour.
        truth=self.coordinates[self.class_labels[:,::4,::4]].permute(0,3,1,2)
        labels=self.labels;codes=self.class_labels
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        decisions=self.nodes['up2_low_dw'][:,:15]
        if not native:
            coord=coord[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
            labels=labels[...,40:-40,40:-40];codes=codes[...,40:-40,40:-40]
            logits=logits[...,20:-20,20:-20];decisions=decisions[...,20:-20,20:-20]
        values=complete_fields(coord/32,truth/32,labels)
        wanted=F.avg_pool2d(self.bit_table[codes].permute(0,3,1,2),2)
        cls=F.binary_cross_entropy_with_logits(logits,wanted)
        shape,sv=formation.formation_loss(p,target,labels);values.update(whole_classification=cls,**sv)
        loss=3*values['whole_field_flat']+2*values['whole_field_colour']+values['whole_field_point']+.5*cls+.2*shape
        if not native and self.step%4==0:
            from .minimum_area_blocks import components,area_merge,boundary_mode
            with torch.no_grad():
                ids=self.predicted_codes(decisions[0]-63.5).cpu().numpy()
                pal=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                rgb=pal[ids];merged,_=area_merge(components(ids),rgb,64)
                clean,_=area_merge(components(boundary_mode(merged,1)),rgb,64)
                count=np.bincount(clean.ravel()).clip(1)
                means=np.stack([np.bincount(clean.ravel(),weights=rgb[...,c].ravel(),minlength=len(count))/count for c in range(3)],1)
                chosen=((means[:,None]-pal[None].astype(float))**2).sum(2).argmin(1)[clean]
                correction=torch.from_numpy(chosen.astype(np.int64)).to(logits.device)[None]
                changed=(correction!=torch.from_numpy(ids).to(logits.device)[None]).float()[:,None]
                bits=self.bit_table[correction].permute(0,3,1,2)
            cleanup=(F.binary_cross_entropy_with_logits(logits[:1],bits,reduction='none')*changed).sum()/(15*changed.sum()).clamp_min(1)
            loss+=.7*cleanup;values.update(actual_island_correction=cleanup,actual_island_changed_fraction=changed.mean())
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(complete_region_supervision=True,area_correction_training_only=True)
        torch.save(state,path)


def regression():
    from .train_b1_flat_colour_fields import field_terms
    lab=torch.arange(32)[None,None,:].expand(1,32,32).long()
    target=(lab[:,None]%3).expand(-1,3,-1,-1).float()
    generator=torch.Generator().manual_seed(61)
    noisy=target+torch.randn(target.shape,generator=generator)*.3
    old=field_terms(noisy,target,lab);new=complete_fields(noisy,target,lab)
    exact=complete_fields(target,target,lab);constant=complete_fields(target*0+1,target,lab)
    assert old['field_flat']==0 and new['whole_field_flat']>.15
    assert all(v<1e-6 for v in exact.values()) and constant['whole_field_colour']>.5
    p=noisy.clone().requires_grad_();sum(complete_fields(p,target,lab).values()).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    print('COMPLETE_REGION_REGRESSION',dict(old_flat=float(old['field_flat']),new_flat=float(new['whole_field_flat']),gradient_nonzero=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    driver.main(training_factory=CompleteRegionTraining,extra_sources=[Path(__file__),here/'train_b1_flat_colour_fields.py',
        Path(ordered.__file__),here/'train_b1_binary_regions.py',here/'build_b1_ordered_palette.py',
        here/'minimum_area_blocks.py',here/'train_b1_clean_regions.py',here/'train_b1_quantum_regions.py',
        here/'fit_b1_sixteen_regions.py',here/'train_b1_sixteen_regions.py',Path(formation.__file__)],
        evaluation=ordered.evaluate,preview=ordered.preview)
