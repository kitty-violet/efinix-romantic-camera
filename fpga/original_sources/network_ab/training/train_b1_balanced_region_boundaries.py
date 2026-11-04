"""Preserve colour while learning independently aligned region boundaries.

Positive/negative examples are balanced per bit, with a4x weight cap. Strong
RGB supervision prevents an easy neutral-grey solution from winning merely
because chromatic foregrounds occupy fewer training pixels.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_independent_region_boundaries import IndependentBoundaryTraining,regression
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def balanced_weight(truth):
    positive=(truth>=.5).float();rate=positive.mean((0,2,3),keepdim=True).clamp(.125,.875)
    return positive*(.5/rate)+(1-positive)*(.5/(1-rate))


def colour_appearance(p,t,mode):
    pixel=F.l1_loss(p,t);coarse=F.l1_loss(F.avg_pool2d(p,8),F.avg_pool2d(t,8))
    return 2*pixel+coarse,dict(palette_colour_rgb=pixel,palette_colour_coarse=coarse)


class BalancedBoundaryTraining(IndependentBoundaryTraining):
    appearance_loss=staticmethod(colour_appearance)

    def distance_loss(self,raw,truth):
        weight=balanced_weight((truth>0).float())
        return (F.smooth_l1_loss(raw,truth,beta=1.,reduction='none')*weight).sum()/weight.sum()

    def classification_loss(self,logits,wanted):
        weight=balanced_weight(wanted)
        return (F.binary_cross_entropy_with_logits(logits,wanted,reduction='none')*weight).sum()/weight.sum()

    def update(self,native=None):
        row=super().update(native)
        with torch.no_grad():
            for key,w in self.net.weight.items():
                if not w.requires_grad:continue
                limit=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
                w.copy_(w.maximum(-limit).minimum(limit))
        return row

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(balanced_region_boundaries=True,trainable_weight_projection=True)
        torch.save(s,path)


def balanced_regression():
    wanted=torch.zeros(1,1,10,10);wanted[:,:,:3]=1
    weight=balanced_weight(wanted)
    positive=float((weight*wanted).sum());negative=float((weight*(1-wanted)).sum())
    assert abs(positive-negative)<1e-4
    perfect=torch.where(wanted>0,torch.tensor(4.),torch.tensor(-4.))
    collapsed=torch.ones_like(perfect)*4
    loss=lambda x:float((F.binary_cross_entropy_with_logits(x,wanted,reduction='none')*weight).sum()/weight.sum())
    assert loss(collapsed)>loss(perfect)+1
    print('BALANCED_BOUNDARY_REGRESSION',dict(positive_mass=positive,negative_mass=negative,
        correct_loss=loss(perfect),collapsed_loss=loss(collapsed)),flush=True)


if __name__=='__main__':
    regression();balanced_regression();here=Path(__file__).parent
    sources=[Path(__file__),here/'train_b1_independent_region_boundaries.py',here/'train_b1_binary_regions.py',
        here/'train_b1_ordered_regions.py',here/'build_b1_ordered_palette.py',here/'train_b1_flat_colour_fields.py',
        here/'minimum_area_blocks.py',here/'train_b1_clean_regions.py',here/'train_b1_quantum_regions.py',
        here/'fit_b1_sixteen_regions.py',here/'train_b1_sixteen_regions.py',here/'train_b1_region_formation.py']
    driver.main(training_factory=BalancedBoundaryTraining,extra_sources=sources,evaluation=evaluate,preview=preview)
