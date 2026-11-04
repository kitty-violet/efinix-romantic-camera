"""Actual B1 training with finite checks for geometry, gamut and flatness.

Every proposed update is evaluated on its own unchanged TRAIN batch. Reject
or shorten updates that sacrifice any of the three appearance objectives.
No validation-based updates, target-dependent inference, or new FPGA ops.
Stability gradients remain in the primary task; video still needs validation.
"""
from pathlib import Path
import torch
from .train_b1_narrow_colour_guard import NarrowColourGuardTraining,evaluate
from .train_b1_narrow_recall_native import objective as geometry
from .b1_local_rgb_envelope import local_envelope,core_error_variation
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .b1_guarded_adam_step import GuardedStepAdam,regression
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


def parameters(net):
    return [v for k,w in net.weight.items() for v in (w,net.bias[k])]


def three_terms(p,t,lab):
    primary,values=geometry(p,t,lab)
    colour,details=local_envelope(p,t,lab);core=core_error_variation(p,t,lab)
    values.update(details);values['rgb_core_error_variation']=core
    return [primary,32*colour,16*core],values


class VerifiedStructuralTraining(NarrowColourGuardTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.opt.__class__=GuardedStepAdam

    def additional_loss(self,p,t,native):
        lab=self.labels if native else self.labels[...,40:-40,40:-40]
        terms,values=three_terms(p,t,lab)
        self.opt.task_gradients=[tuple(g.detach() for g in torch.autograd.grad(v,parameters(self.net),retain_graph=True)) for v in terms[1:]]
        initial=torch.stack(terms).detach()
        def check():
            q=self.net(self.last_source)[0]
            if not native:q=q[...,40:-40,40:-40]
            return torch.stack(three_terms(q,t,lab)[0])
        self.opt.finite_check=(initial,check)
        with torch.no_grad():values['complete_rgb_profile']=complete_rgb_profiles(p,t,lab)[0]
        return sum(terms),values

    def update(self,native=None):
        result=super().update(native)
        result['finite_appearance_check']=self.opt.last_finite_check
        result['task_step_projection']=self.opt.last_projection
        return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(verified_structural_steps=dict(tasks=['geometry','RGB_envelope','core_variation'],
            finite_check='same TRAIN batch all3 nonincreasing tolerance1e-7',
            step_fractions=[1,.5,.25,.125,.0625,0],optimizer_moments_retained_after_rejection=True,
            stability_gradients_retained=True,stability_not_finite_checked=True,
            extra_inference_operations=0),visual_acceptance=False)
        torch.save(state,path)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=VerifiedStructuralTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
