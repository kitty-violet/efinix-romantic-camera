"""Keep auxiliary contour training from overriding the actual RGB objective.

Seven TRAIN crops exposed conflicts after the encoder Jacobian that a field-
only norm calibration missed. Project a conflicting auxiliary gradient away
from the RGB gradient and cap its norm at half the RGB norm. This is an
asymmetric, bounded adaptation of PCGrad, not the published PCGrad algorithm.
https://arxiv.org/abs/2001.06782

This is a raw-gradient property, not a monotonicity guarantee for Adam, QAT or
held-out images. Actual exported-image and video checks remain mandatory.
"""
from pathlib import Path
import torch
from .train_b1_coincident_distance import CoincidentDistanceTraining,coincident_loss,evaluate,driver,preview
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_ordered_boundary_fields import rank_loss


def protected_auxiliary(rgb,aux,limit=.5):
    rr=(rgb*rgb).sum();dot=(rgb*aux).sum()
    projected=aux-torch.minimum(dot,dot.new_zeros(()))/rr.clamp_min(1e-20)*rgb
    cap=(limit*rr.sqrt()/projected.norm().clamp_min(1e-20)).clamp_max(1)
    if rr<1e-20:cap=cap*0
    return projected*cap,dict(raw_cosine=dot/(rgb.norm()*aux.norm()).clamp_min(1e-20),
        raw_norm_ratio=aux.norm()/rgb.norm().clamp_min(1e-20),auxiliary_cap=cap)


def gradient_correction(rgb_loss,aux_loss,parameters):
    gr=torch.autograd.grad(rgb_loss,parameters,retain_graph=True,allow_unused=True)
    ga=torch.autograd.grad(aux_loss,parameters,retain_graph=True,allow_unused=True)
    gr=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,gr)])
    ga=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,ga)])
    adjusted,values=protected_auxiliary(gr,ga)
    correction=(adjusted-ga).detach();offset=0;surrogate=rgb_loss.new_zeros(())
    for p in parameters:
        surrogate=surrogate+(p.flatten()*correction[offset:offset+p.numel()]).sum();offset+=p.numel()
    assert offset==len(correction)
    values.update(protected_auxiliary_cosine=(gr*adjusted).sum()/(gr.norm()*adjusted.norm()).clamp_min(1e-20),
        protected_auxiliary_norm_ratio=adjusted.norm()/gr.norm().clamp_min(1e-20))
    return surrogate-surrogate.detach(),{k:v.detach() for k,v in values.items()}


class RGBPriorityDistanceTraining(CoincidentDistanceTraining):
    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        field=self.nodes['up1_low_dw'];truth=self.supervision_distance_truth
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        full,band=distance_objective(field,truth)
        auxiliary=self.args.classification*(8*full+4*band)+10*rank_loss(field)+32*coincident_loss(field,truth)
        rgb=loss-auxiliary
        correction,details=gradient_correction(rgb,auxiliary,[p for p in self.net.parameters() if p.requires_grad])
        values.update(details)
        return loss+correction,values

    def preflight_extras(self):
        row=super().preflight_extras()
        assert row['values']['protected_auxiliary_cosine']>=-1e-5
        assert row['values']['protected_auxiliary_norm_ratio']<=.50001
        return row

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(rgb_priority_gradient=True,auxiliary_gradient_norm_limit=.5,
            gradient_projection_scope='raw trainable encoder parameters before clipping and RowScaledAdam',
            raw_rgb_gradient_preserved=True,adam_monotonicity_guaranteed=False,added_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)


def regression():
    for r,a in [(torch.tensor([1.,0.]),torch.tensor([-4.,3.])),
                (torch.tensor([1.,0.]),torch.tensor([4.,3.])),
                (torch.zeros(2),torch.ones(2)),(torch.ones(2),torch.zeros(2))]:
        q,_=protected_auxiliary(r,a)
        assert (r*q).sum()>=-1e-7 and q.norm()<=.500001*r.norm()
        assert torch.isfinite(q).all()
    p=torch.nn.Parameter(torch.tensor([.2,.4]));rgb=(p*torch.tensor([1.,0.])).sum();aux=(p*torch.tensor([-4.,3.])).sum()
    correction,v=gradient_correction(rgb,aux,[p]);assert correction==0
    (rgb+aux+correction).backward();assert torch.allclose(p.grad,torch.tensor([1.,.5]),atol=1e-6)
    print('RGB_PRIORITY_GRADIENT_PASS conflict_projection/norm_cap/zero_norm/forward_unchanged/actual_gradient',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=RGBPriorityDistanceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
