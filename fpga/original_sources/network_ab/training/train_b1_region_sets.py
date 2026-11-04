"""Teach complete colour-region membership, not mean RGB reconstruction.

The vectorized symmetric Lovasz hinge below adapts the binary formulation of
Berman et al. (CVPR2018), https://github.com/bermanmaxim/LovaszSoftmax .
It covers BOTH a bit's foreground and background, including absent classes,
so isolated false colours cannot disappear inside a large pixelwise average.

MIT License -- Copyright (c) 2018 Maxim Berman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_native_distance_boundaries import NativeDistanceTraining
from .train_b1_balanced_region_boundaries import balanced_weight
from .train_b1_decisive_regions import modal_region_cleanup
from .train_b1_active_colour_boundaries import active_colour_boundary_loss
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def symmetric_region_hinge(logits,truth):
    valid=(truth<.01)|(truth>.99);labels=(truth>=.5).float()
    score=torch.cat((logits,-logits),1).flatten(0,1).flatten(1)
    label=torch.cat((labels,1-labels),1).flatten(0,1).flatten(1)
    valid=torch.cat((valid,valid),1).flatten(0,1).flatten(1)
    label=torch.where(valid,label,0.)
    error=torch.where(valid,1-score*(2*label-1),float('-inf'))
    ranked,index=error.sort(dim=1,descending=True,stable=True)
    foreground=label.gather(1,index).detach();total=foreground.sum(1,keepdim=True)
    intersection=total-foreground.cumsum(1)
    union=total+(1-foreground).cumsum(1)
    jaccard=1-intersection/union.clamp_min(1)
    increment=torch.cat((jaccard[:,:1],jaccard[:,1:]-jaccard[:,:-1]),1)
    return (ranked.relu()*increment).sum(1).mean()


def coarse_colour(p,t,mode):
    loss=F.l1_loss(F.avg_pool2d(p,8),F.avg_pool2d(t,8))
    return .5*loss,dict(coarse_colour_anchor=loss)


class RegionSetTraining(NativeDistanceTraining):
    appearance_loss=staticmethod(coarse_colour)

    def additional_loss(self,p,target,native):
        full=self.bit_table[self.class_labels].permute(0,3,1,2)
        wanted=F.avg_pool2d(full,2);distance=self.distance_targets(full)
        key='up1_low_pw';s=(getattr(self.net,key+'_m')/131072).float().flatten()
        if self.net.quantized:
            w=self.net.round(self.net.weight[key]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
            b=self.net.round(self.net.bias[key]/s)*s
        else:
            # A floating-learning control must not silently keep quantized
            # coefficients in its auxiliary distance supervision.
            w=self.net.weight[key];b=self.net.bias[key]
        raw=(F.conv2d(self.nodes['up1_low_dw'],w,b)[:,:15]-63.5)/8
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        codes=self.class_labels[:,::2,::2]
        hard=self.nodes['up2_low_dw'][:,:15]
        if not native:
            raw=raw[...,10:-10,10:-10];distance=distance[...,10:-10,10:-10]
            logits=logits[...,20:-20,20:-20];wanted=wanted[...,20:-20,20:-20]
            codes=codes[...,20:-20,20:-20];hard=hard[...,20:-20,20:-20]
        overlap=symmetric_region_hinge(logits,wanted)
        weight=balanced_weight(wanted)
        ce=(F.binary_cross_entropy_with_logits(logits,wanted,reduction='none')*weight).sum()/weight.sum()
        phase=F.smooth_l1_loss(raw,distance,beta=1.)
        loss=overlap+.5*ce+.25*phase
        values=dict(region_set_hinge=overlap,region_set_ce=ce,native_distance_anchor=phase)
        if not native:
            actual=torch.stack([self.predicted_codes(v-63.5) for v in hard])
            direction,details=active_colour_boundary_loss(logits,actual,codes)
            loss+=.25*direction;values.update(details)
            with torch.no_grad():
                ids=self.predicted_codes(self.nodes['up2_low_dw'][0,:15]-63.5).cpu().numpy()
                pal=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                clean,_=modal_region_cleanup(ids,pal,64,2)
                clean=torch.tensor(clean[20:-20,20:-20],device=p.device)[None]
                # A self-projection cannot override a different labelled colour.
                change=((clean!=actual[:1])&(clean==codes[:1])).float()[:,None]
                desired=self.bit_table[clean].permute(0,3,1,2)
            correction=(F.binary_cross_entropy_with_logits(logits[:1],desired,reduction='none')*change).sum()/(15*change.sum()).clamp_min(1)
            loss+=correction;values.update(consistent_area_correction=correction,
                consistent_area_fraction=change.mean())
        return loss,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(region_set_objective=True,symmetric_lovasz_bits=True,
            objective='set hinge1 + balanced CE.5 + native SDF.25 + active boundary.25 + consistent modal area1 + coarse RGB.5',
            full_rgb_pixel_regression=False,minimum_area_inference_guarantee=False)
        torch.save(s,path)


def regression():
    result={}
    for size in (32,128):
        target=torch.zeros(1,1,size,size);good=torch.full_like(target,-3.)
        bad=good.clone();bad[0,0,size//2,size//2]=3.;bad.requires_grad_(True)
        exact=symmetric_region_hinge(good,target);junk=symmetric_region_hinge(bad,target)
        junk.backward();assert float(exact)==0 and float(junk.detach())>.5
        assert float(bad.grad[0,0,size//2,size//2])>0
        result[size]=float(junk.detach())
    assert abs(result[32]-result[128])<.01,'A lone false colour must not be averaged away'
    target=torch.zeros(1,1,32,32);target[:,:,12:20,12:20]=1
    exact=(target*2-1)*3;missing=torch.full_like(exact,-3)
    assert float(symmetric_region_hinge(exact,target))==0
    assert float(symmetric_region_hinge(missing,target))>1
    print('REGION_SET_REGRESSION',result,'missing_object_penalized',flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_native_distance_boundaries.py','train_b1_active_colour_boundaries.py',
        'train_b1_decisive_regions.py','probe_b1_independent_decoder.py',
        'train_b1_independent_region_boundaries.py','train_b1_balanced_region_boundaries.py',
        'train_b1_binary_regions.py','train_b1_ordered_regions.py','build_b1_ordered_palette.py',
        'minimum_area_blocks.py','train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=RegionSetTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
