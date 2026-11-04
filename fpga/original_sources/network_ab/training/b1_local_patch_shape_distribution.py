"""Local distributions of complete RGB patch shapes, training only.

Method inspiration: Elnekave & Weiss, ECCV2022, arxiv.org/abs/2203.11862
and github.com/ariel415el/GPDM/blob/main/patch_swd.py (Apache-2.0).
This independently implemented variant uses fixed projections, per-channel
zero-DC filters and local32x32 windows; it is not their global image optimizer.
"""
from functools import lru_cache
import torch
from torch.nn import functional as F


@lru_cache(maxsize=1)
def cpu_filters():
    g=torch.Generator(device='cpu').manual_seed(2026100793)
    bank=torch.randn(64,3,7,7,generator=g)
    bank=bank-bank.mean((-2,-1),keepdim=True)
    return bank/bank.flatten(1).norm(dim=1)[:,None,None,None]


def patch_shape_distribution(p,t,cell=32):
    assert p.shape==t.shape and p.shape[-2]>=38 and p.shape[-1]>=38
    kernel=cpu_filters().to(device=p.device,dtype=p.dtype)
    a=F.conv2d(p,kernel);b=F.conv2d(t,kernel)
    hh=a.shape[-2]//cell*cell;ww=a.shape[-1]//cell*cell
    def windows(z):
        return z[...,:hh,:ww].unfold(2,cell,cell).unfold(3,cell,cell).flatten(-2).sort(-1).values
    # Equal window/sample counts; no random duplication, image reshuffling,
    # or change to the output is performed.
    return (windows(a)-windows(b)).abs().mean()


def regression():
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(96),indexing='ij')
    target=torch.where(xx>46+.3*yy,-.5,.5)[None,None].expand(1,3,-1,-1).clone()
    assert patch_shape_distribution(target,target)==0
    shifted_colour=target+torch.tensor([.12,-.05,.07])[None,:,None,None]
    assert patch_shape_distribution(shifted_colour,target)<1e-6
    blur=F.avg_pool2d(F.pad(target,(4,4,4,4),mode='replicate'),9,1).requires_grad_()
    loss=patch_shape_distribution(blur,target);g=torch.autograd.grad(loss,blur)[0]
    assert loss>.01 and (g*(target-blur.detach())).sum()<0 and torch.isfinite(g).all()
    # Equal pixel histograms but incompatible edge arrangement must differ.
    solid=torch.where(xx<48,-.5,.5)[None,None].expand(1,3,-1,-1).clone()
    checker=torch.where((xx+yy)%2==0,-.5,.5)[None,None].expand_as(solid).clone()
    assert torch.equal(solid.flatten().sort().values,checker.flatten().sort().values)
    assert patch_shape_distribution(checker,solid)>.05
    print('PATCH_SHAPE_PASS target0/DC_invariance/blur_direction/finite/same_colours_wrong_shapes_rejected',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
