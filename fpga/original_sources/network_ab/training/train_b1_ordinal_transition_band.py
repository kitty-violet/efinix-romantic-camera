"""Constrain the full ordinal transition band, including both shoulders.

Supervising only fractional target cells leaves the two surrounding pure sides
to the diluted whole-frame loss. This term targets all changing ordinal fields
within a20px native band. It cannot replace colour, shape or video review.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_direct_ordinal_pairs import DirectOrdinalPairTraining,ordinal_loss,driver,preview,rgb_evaluate
from .build_b1_thirty_binary_regions import code_bits

def transition_band_loss(field,truth):
    high=F.max_pool2d(truth,5,1,2);low=-F.max_pool2d(-truth,5,1,2)
    band=(high-low)>.02
    terms=F.smooth_l1_loss(field[:,:30]/127,truth,beta=.08,reduction='none')
    # Keep the many unchanged colour predicates outside this normalization.
    return (terms*band).sum()/band.sum().clamp_min(1)

class TransitionBandTraining(DirectOrdinalPairTraining):
    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        field=self.nodes['up1_low_dw'];truth=F.avg_pool2d(code_bits(self.class_labels),4)
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        band=transition_band_loss(field,truth)
        physical=F.smooth_l1_loss(field[:,:30]/127,truth,beta=.08)
        # The boundary band already focuses rare contours. Do not additionally
        # overweight the area of rare positive predicates in whole-frame loss.
        loss+=self.args.classification*(4*band+8*physical-8*v['ordered_area']-2*v['ordered_boundary_area'])
        v.update(ordered_full_transition_band=band,physical_ordinal_area=physical)
        return loss,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(ordinal_transition_band_native=20,fractional_cell_loss_replaced=True,
          ordinal_area_class_rebalancing=False,visual_acceptance=False);torch.save(s,path)

def regression():
    t=torch.zeros(1,30,16,32);t[...,16]=.5;t[...,17:]=1
    narrow=t*127;wide=narrow.clone();wide[...,14]=.15*127;wide[...,15]=.3*127;wide[...,17]=.7*127;wide[...,18]=.85*127
    _,old=ordinal_loss(wide,t);assert old==0
    assert transition_band_loss(narrow,t)==0
    wide.requires_grad_();new=transition_band_loss(wide,t);assert new>.04;new.backward()
    assert wide.grad[...,14].mean()>0 and wide.grad[...,18].mean()<0 and torch.isfinite(wide.grad).all()
    plain=torch.zeros_like(t);assert transition_band_loss(plain,plain)==0
    print('ORDINAL_TRANSITION_BAND_PASS exact0/mixed_only_misses_shoulders/full_band_detects/directional_gradient/empty0',float(new.detach()),flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:driver.main(training_factory=TransitionBandTraining,extra_sources=[Path(__file__)],evaluation=rgb_evaluate,preview=preview)
