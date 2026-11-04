"""Proper soft classification loss at B1's actual last boundary decision.

The 30 supervised distance fields are combined before the15 final ramps.
Distance errors far from a decision need not change the rendered colour.
Use the exact known-target decoder's pre-ramp field as a soft logit target,
with a Bernoulli KL that has its minimum at that target. Training only.
"""
import torch
from torch.nn import functional as F
from .b1_binary_layout import RADIX_WEIGHTS
from .build_b1_thirty_binary_regions import GROUPS30


def smooth_rank_loss(z):
    violation=torch.cat([F.relu(torch.diff(z[:,a:b],dim=1)) for a,b in GROUPS30],1).amax(1)/127
    return violation.square().mean()


def rendering_decision_loss(prediction,target,temperature=3.):
    p=prediction[:,:15];t=target[:,:15].detach()
    pp=(p-63.5)/temperature;tt=(t-63.5)/temperature
    # KL(Bernoulli(sigmoid(tt)) || Bernoulli(sigmoid(pp))). Stable in the
    # legal0..127 range. Temperature/127 keeps wrong-side gradients in code
    # units comparable to normalized distance L1, without a saturated tail.
    # Default softplus switches to a linear branch at20; B1 logits reach
    #21.17 here. Keep the true softplus derivative over the complete range.
    terms=(F.softplus(pp,threshold=64)-F.softplus(tt,threshold=64)-(pp-tt)*torch.sigmoid(tt)).clamp_min(0)*(temperature/127)
    weights=p.new_tensor([v for group in RADIX_WEIGHTS for v in group])[None,:,None,None]
    full=(terms*weights).sum()/(weights.sum()*p.shape[0]*p.shape[2]*p.shape[3])
    # Fixed target support: the predictor cannot dilute this loss merely by
    # inventing many additional uncertain pixels. False edges still incur full.
    band=(t-63.5).abs()<24
    weighted=weights*band
    near=(terms*weighted).sum()/weighted.sum().clamp_min(1)
    return full,near


def regression():
    t=torch.linspace(0,127,128,dtype=torch.float64)[None,None,None].expand(1,15,4,128).clone()
    p=t.clone().requires_grad_();loss=sum(rendering_decision_loss(p,t));assert loss==0
    loss.backward();assert p.grad.abs().max()<1e-15
    p=(127-t).clone().requires_grad_();loss=sum(rendering_decision_loss(p,t));assert loss>.1;loss.backward()
    assert p.grad[...,0].min()>0 and p.grad[...,-1].max()<0 and torch.isfinite(p.grad).all()
    # Same four-code displacement: correct saturated tail is less important
    # than the field at the actual colour switch, with both still detectable.
    tt=torch.tensor([63.5,120.],dtype=torch.float64)[None,None,None].expand(1,15,1,2)
    pp=(tt+4).clone().requires_grad_();loss=sum(rendering_decision_loss(pp,tt));loss.backward()
    assert pp.grad[...,0].abs().mean()>1000*pp.grad[...,1].abs().mean()
    pt=torch.linspace(5,120,15,dtype=torch.float64)[None,:,None,None].expand(1,15,2,2).clone()
    pp=(pt+2.3).clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda z:sum(rendering_decision_loss(z,pt)),(pp,),atol=1e-5,rtol=1e-3)
    pp=pt.float().clone().requires_grad_();fp=sum(rendering_decision_loss(pp,pt.float()));assert fp==0
    fp.backward();assert pp.grad.abs().max()<1e-8
    z=torch.zeros(1,32,4,4,dtype=torch.float64);assert smooth_rank_loss(z)==0
    z[:,3]=1;z.requires_grad_();rv=smooth_rank_loss(z);rv.backward()
    assert z.grad[:,3].sum()>0 and z.grad[:,2].sum()<0
    assert abs(float(rv.detach())-1/127**2)<1e-12
    print('RENDER_BOUNDARY_LOSS_PASS target0/zero_gradient/wrong_side_direction/decision_sensitivity/gradcheck/FP32/squared_rank',flush=True)


if __name__=='__main__':regression()
