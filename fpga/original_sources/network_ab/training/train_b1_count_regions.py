"""Supervise actual palette levels, not arbitrary ordered internal bit codes.

The deployed balanced decoder counts7/4/4 independent binary fields. Any
permutation within a group has identical RGB, so canonical bit BCE/SDF imposes
constraints that are not part of the output function. A log-space Bernoulli
count distribution supervises the actual8/5/5 levels instead. Probability DP
and connected-area correction exist in training only; export stays plain B1.
"""
from pathlib import Path
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_native_distance_boundaries import NativeDistanceTraining
from .train_b1_decisive_regions import modal_region_cleanup
from .train_b1_ordered_regions import evaluate,preview
from .build_b1_ordered_palette import GROUPS
from . import train_b1_late_region_supervision as driver


def log_count_probability(logits):
    dist=torch.zeros_like(logits[:,:1])
    for bit in logits.split(1,dim=1):
        off=dist+F.logsigmoid(-bit);on=dist+F.logsigmoid(bit)
        impossible=torch.full_like(off[:,:1],float('-inf'))
        dist=torch.logaddexp(torch.cat((off,impossible),1),torch.cat((impossible,on),1))
    return dist


def colour_levels(codes):return (codes//25,(codes//5)%5,codes%5)


def count_targets(codes,scale=2):
    return [F.avg_pool2d(F.one_hot(level,n).permute(0,3,1,2).float(),scale)
        for level,n in zip(colour_levels(codes),(8,5,5))]


def count_loss(logits,targets,mask=None,balanced=True):
    losses=[]
    for (start,end),truth in zip(GROUPS,targets):
        logp=log_count_probability(logits[:,start:end])
        if balanced:
            freq=truth.mean((0,2,3),keepdim=True)
            weight=freq.clamp_min(1/(4*truth.shape[1])).rsqrt()
            weight=(weight/(freq*weight).sum()).clamp(.25,4).detach()
        else:weight=torch.ones_like(truth[:1,:,:1,:1])
        weighted=truth*weight
        if mask is not None:weighted=weighted*mask
        losses.append(-(weighted*logp).sum()/weighted.sum().clamp_min(1))
    return torch.stack(losses).mean()


class CountRegionTraining(NativeDistanceTraining):
    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def additional_loss(self,p,target,native):
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        truth=count_targets(self.class_labels)
        if not native:
            logits=logits[...,20:-20,20:-20];truth=[v[...,20:-20,20:-20] for v in truth]
        supervised=count_loss(logits,truth)
        values=dict(actual_palette_count_ce=supervised)
        if native:return supervised,values
        with torch.no_grad():
            palette=(self.palette.detach().cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
            desired=[];changed=[]
            for item in self.nodes['up2_low_dw'][:,:15]:
                code=self.predicted_codes(item-63.5).cpu().numpy()
                clean,_=modal_region_cleanup(code,palette,64,2)
                desired.append(torch.tensor(clean[20:-20,20:-20],device=p.device))
                changed.append(torch.tensor((clean!=code)[20:-20,20:-20],device=p.device))
            desired=count_targets(torch.stack(desired),1);mask=torch.stack(changed)[:,None].float()
        correction=count_loss(logits,desired,mask,balanced=False)
        values.update(actual_count_area_correction=correction,projected_changed_fraction=mask.mean())
        return supervised+.3*correction,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(actual_palette_count_supervision=True,canonical_bit_supervision=False,
            canonical_distance_supervision=False,signed_distance_training_only=False,
            minimum_area_inference_guarantee=False,
            objective='balanced count CE1 + changed-only count projection.3 + existing temporal stability',
            inference_added_operators=0)
        torch.save(s,path)


def regression():
    torch.manual_seed(37)
    z=torch.randn(2,7,2,3,dtype=torch.float64,requires_grad=True)
    dist=log_count_probability(z).exp();brute=torch.zeros_like(dist)
    for pattern in itertools.product((0,1),repeat=7):
        b=z.new_tensor(pattern)[None,:,None,None]
        probability=torch.where(b>0,z.sigmoid(),(-z).sigmoid()).prod(1)
        brute[:,sum(pattern)]+=probability
    error=float((dist-brute).abs().max().detach());assert error<1e-12
    perm=log_count_probability(z[:,[3,6,0,4,2,5,1]])
    assert torch.allclose(dist.log(),perm,atol=1e-12,rtol=0)
    assert torch.autograd.gradcheck(log_count_probability,(z[:1,:,0:1,0:1],),eps=1e-6,atol=1e-5)
    # Identical level=3 with two very different internal field assignments.
    a=torch.tensor([8,8,8,-8,-8,-8,-8],dtype=torch.float64)[None,:,None,None]
    b=a[:,[3,4,5,0,1,2,6]]
    proper_a=-log_count_probability(a)[:,3].mean();proper_b=-log_count_probability(b)[:,3].mean()
    old=F.binary_cross_entropy_with_logits(b,(a>0).double())
    assert abs(float(proper_a-proper_b))<1e-12 and float(old)>6 and float(proper_b)<.003
    low=torch.full((1,7,1,1),-5.,requires_grad=True)
    (-log_count_probability(low)[:,3].mean()).backward();assert low.grad.max()<0
    print('COUNT_LIKELIHOOD_REGRESSION',dict(brute_128_patterns_max_error=error,permutation_invariant=True,
        same_colour_count_loss=float(proper_b),same_colour_old_bit_loss=float(old),gradient_check=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_native_distance_boundaries.py','train_b1_decisive_regions.py',
        'probe_b1_independent_decoder.py','train_b1_independent_region_boundaries.py',
        'train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py',
        'train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py',
        'train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=CountRegionTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
