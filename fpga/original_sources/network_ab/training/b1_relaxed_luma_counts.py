"""Joint15-level count likelihood with native boundary-label relaxation.

The14 positive Y contributions already exist. Sum their independent binary
decision probabilities into a Poisson-binomial count distribution during
TRAINING only. This prevents confusing14 marginal decisions with a jointly
confident region choice. Nearby allowed labels are a set, not their average.
Boundary-label-set idea: https://arxiv.org/abs/1812.01593 ; adapted to B1
brightness counts, not a reproduction of that segmentation architecture.
"""
import torch
from torch.nn import functional as F


def count_log_probability(field,temperature=3.):
    logits=(field-63.5)/temperature
    yes=F.logsigmoid(logits);no=F.logsigmoid(-logits)
    dp=torch.zeros_like(field[:,:1])
    for k in range(field.shape[1]):
        stay=F.pad(dp+no[:,k:k+1],(0,0,0,0,0,1),value=-1e4)
        rise=F.pad(dp+yes[:,k:k+1],(0,0,0,0,1,0),value=-1e4)
        dp=torch.logaddexp(stay,rise)
    return dp


@torch.no_grad()
def allowed_levels(codes720,radius=2):
    classes=codes720//81
    hot=F.one_hot(classes.long(),15).permute(0,3,1,2).float()
    # Each360 cell has a2x2 native footprint plus the declared native margin.
    return F.max_pool2d(hot,2+2*radius,2,radius)>0


def relaxed_count_loss(field,allowed):
    assert field.shape[1]==14 and allowed.shape[1]==15
    assert field.shape[0:1]+field.shape[2:]==allowed.shape[0:1]+allowed.shape[2:]
    assert allowed.any(1).all()
    logp=count_log_probability(field)
    loss=-torch.logsumexp(logp.masked_fill(~allowed,-1e4),1)*(3/127)
    boundary=allowed.sum(1)>1
    return loss.mean(),(loss*boundary).sum()/boundary.sum().clamp_min(1)


def regression():
    # Distribution matches exhaustive enumeration for4 independent bits.
    p=torch.tensor([.13,.27,.62,.89],dtype=torch.float64)
    field=(63.5+3*torch.logit(p))[None,:,None,None]
    got=count_log_probability(field).exp().flatten();want=torch.zeros(5,dtype=torch.float64)
    for mask in range(16):
        prob=1.;n=0
        for k in range(4):prob=prob*(p[k] if mask>>k&1 else 1-p[k]);n+=mask>>k&1
        want[n]+=prob
    assert torch.allclose(got,want,atol=1e-12)
    ff=torch.linspace(15,112,14,dtype=torch.float64)[None,:,None,None].requires_grad_()
    assert torch.autograd.gradcheck(count_log_probability,(ff,),atol=1e-5,rtol=1e-3)
    # Distinguish an endpoint choice from the forbidden middle of two levels.
    valid=torch.zeros(1,15,1,1,dtype=torch.bool);valid[:,2]=True;valid[:,12]=True
    def hard(n):
        q=torch.zeros(1,14,1,1);q[:,:n]=127;return q
    end=sum(relaxed_count_loss(hard(2),valid));middle=sum(relaxed_count_loss(hard(7),valid));uncertain=sum(relaxed_count_loss(torch.full((1,14,1,1),63.5),valid))
    assert end<1e-5 and middle>.2 and uncertain>.05
    # Loss on a one-label interior must push erroneous upper bits downward.
    one=torch.zeros_like(valid);one[:,2]=True;q=hard(7).requires_grad_();v=sum(relaxed_count_loss(q,one));v.backward()
    assert q.grad[:,2:7].min()>0 and torch.isfinite(q.grad).all()
    codes=torch.full((1,32,48),2*81+40,dtype=torch.long);codes[...,24:]=12*81+40
    support=allowed_levels(codes);assert support.shape==(1,15,16,24) and support[:,:,:,12].sum(1).eq(2).all()
    assert support[:,:,:,0].sum(1).eq(1).all() and support[:,7].sum()==0
    print('RELAXED_LUMA_COUNTS_PASS enumeration/gradcheck/endpoint_not_middle/direction/native_support',float(end),float(middle),float(uncertain),flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
