"""Training-only critic gradient projection in RGB image space.

Reuse the gradient-conflict idea of PCGrad (arxiv:2001.06782), here projected
against the actual target displacement, not another parameter gradient.
No guarantee is made for the later B1 Jacobian, quantization or Adam step.
"""
import torch


def project(gradient,direction):
    axes=tuple(range(1,gradient.ndim))
    dot=(gradient*direction).sum(axes,keepdim=True)
    norm=direction.square().sum(axes,keepdim=True)
    # Descent follows -gradient. Positive dot with target-prediction would
    # therefore increase the target error to first order in image space.
    amount=dot.clamp_min(0)/norm.clamp_min(1e-12)
    return gradient-amount*direction


class TargetAlignedCriticInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx,prediction,target):
        ctx.save_for_backward((target-prediction).detach())
        return prediction

    @staticmethod
    def backward(ctx,gradient):
        direction,=ctx.saved_tensors
        return project(gradient,direction),None


def regression():
    torch.manual_seed(71)
    p=torch.randn(3,3,11,13,requires_grad=True);t=torch.randn_like(p);direction=t-p.detach()
    g=torch.randn_like(p);g[0]=direction[0];g[1]=-direction[1]
    q=TargetAlignedCriticInput.apply(p,t);assert torch.equal(p,q)
    (q*g).sum().backward();expected=project(g,direction)
    assert torch.allclose(p.grad,expected)
    assert (p.grad*direction).flatten(1).sum(1).max()<1e-5
    assert torch.equal(p.grad[1],g[1]) and p.grad[0].abs().max()<1e-6
    assert torch.equal(project(g,torch.zeros_like(g)),g)
    assert (p.grad.flatten(1).norm(dim=1)<=g.flatten(1).norm(dim=1)+1e-5).all()
    print('TARGET_ALIGNED_CRITIC_PASS forward_equal/actual_backward/conflict_removed/nonconflict_unchanged/zero/norm',flush=True)


if __name__=='__main__':regression()
