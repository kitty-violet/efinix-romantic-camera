"""Stronger decisive-region critic with a measured target-conflict guard."""
from pathlib import Path
import torch
from .b1_target_aligned_critic import TargetAlignedCriticInput,regression
from .train_b1_local_colour_commitment import LocalColourCommitmentTraining,evaluate
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


class TargetAlignedCriticTraining(LocalColourCommitmentTraining):
    @staticmethod
    def adversarial_input(prediction,target):
        return TargetAlignedCriticInput.apply(prediction,target)

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(critic_target_gradient_projection=dict(scope='per-image critic RGB crop',
            reference='actual stylized target minus current prediction',forward_unchanged=True,
            discriminator_gradients_unchanged=True,guarantees_only='projected image gradient does not oppose target displacement',
            does_not_guarantee='B1 parameter or quantized Adam update improvement'),visual_acceptance=False)
        torch.save(state,path)

    def preflight_extras(self):
        result=super().preflight_extras();assert self.critic is not None
        source=self.last_source[:1,...,40:136,40:136]
        with torch.no_grad():p=self.net(self.last_source[:1])[0][...,40:136,40:136]
        p=p.detach().requires_grad_();t=self.hard_target[:1,...,40:136,40:136]
        q=self.adversarial_input(p,t);assert torch.equal(p,q)
        loss=-torch.stack([v.mean() for v in self.critic(q,source)]).mean()
        grad=torch.autograd.grad(loss,p)[0];dot=float((grad*(t-p.detach())).sum())
        assert torch.isfinite(grad).all() and grad.norm()>0 and dot<1e-5
        result['critic_projection_actual_gpu']=dict(forward_error=0,gradient_dot_target_displacement=dot,gradient_norm=float(grad.norm()))
        return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=TargetAlignedCriticTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
