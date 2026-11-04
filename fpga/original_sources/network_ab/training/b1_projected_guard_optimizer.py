"""Two-task PCGrad in the existing TRAIN-whitened coefficient coordinates.

Geometry plus all existing stability terms form one task; RGB-envelope/core
cleanliness form the other. Only opposing components are removed. Inference
is unchanged. The common-descent statement applies to these gradient vectors,
NOT to arbitrary finite, quantized, momentum-Adam steps; verify those separately.
Reference: Yu et al., NeurIPS2020, https://arxiv.org/abs/2001.06782
"""
import torch
from .whitened_b1_optimizer import WhitenedB1Adam,transform


def project_two(primary,guard):
    dot=(primary.double()*guard.double()).sum()
    p2=primary.double().square().sum();g2=guard.double().square().sum()
    cosine=dot/(p2*g2).sqrt().clamp_min(1e-30)
    if dot<0:
        pp=primary-(dot/g2.clamp_min(1e-30)).to(primary.dtype)*guard
        gg=guard-(dot/p2.clamp_min(1e-30)).to(primary.dtype)*primary
    else:pp,gg=primary,guard
    combined=pp+gg
    return combined,dict(cosine=float(cosine),projected=bool(dot<0),
        primary_dot_combined=float((primary.double()*combined.double()).sum()),
        guard_dot_combined=float((guard.double()*combined.double()).sum()),
        norm_ratio=float(combined.norm()/(primary+guard).norm().clamp_min(1e-20)))


class ProjectedGuardAdam(WhitenedB1Adam):
    def prepare_gradients(self):
        assert self.guard_gradients is not None
        raw=self.guard_gradients;guard=[]
        for i,k in enumerate(self.net.weight):
            gw,gb=raw[2*i:2*i+2];group=self.param_groups[2*i]
            assert group['node_key']==k
            ga=transform(gw.flatten(1)-gb[:,None]*group['patch_mean'].to(gw.device),
                         group['input_whitener'].to(gw.device).transpose(-1,-2)).reshape_as(gw)
            guard.extend((ga,gb))
        super().prepare_gradients()
        parameters=[g['params'][0] for g in self.param_groups]
        total=torch.cat([p.grad.flatten() for p in parameters])
        g=torch.cat([v.flatten() for v in guard])
        adjusted,self.last_projection=project_two(total-g,g)
        offset=0
        for p in parameters:
            p.grad.copy_(adjusted[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
        assert offset==adjusted.numel()
        self.guard_gradients=None


def regression():
    gen=torch.Generator().manual_seed(190)
    for _ in range(100):
        a=torch.randn(97,generator=gen);b=torch.randn(97,generator=gen)*.3
        for v in (b,a*0,-a,a):
            result,stats=project_two(a,v)
            assert torch.isfinite(result).all()
            assert stats['primary_dot_combined']>=-1e-4 and stats['guard_dot_combined']>=-1e-4
            if (a*v).sum()>=0:assert torch.equal(result,a+v)
    print('PROJECT_TWO_PASS aligned/orthogonal/conflicting/opposite/zero/common_gradient_descent',flush=True)


if __name__=='__main__':regression()
