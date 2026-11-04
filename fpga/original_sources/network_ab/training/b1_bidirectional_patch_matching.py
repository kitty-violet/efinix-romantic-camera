"""Bounded bidirectional RGB patch correspondence, training only.

Unlike one-way contour profiles, both generated-patch coherence and target
coverage contribute. One whole 9x9 RGB patch uses one 2D offset; its channels
cannot match independently. Matching is detached; the selected exact L1
distances retain gradients. No RGB patch is pasted or filtered at inference.
Related non-parametric patch principle: Li & Wand, CVPR2016, arxiv:1601.04589.
This operates on RGB in a bounded paired neighbourhood, not their VGG MRF.
"""
import torch
from torch.nn import functional as F


def points(target,count):
    n,_,h,w=target.shape;margin=13
    mask=torch.ones(n,h,w,dtype=torch.bool,device=target.device)
    mask[:,:margin]=False;mask[:,-margin:]=False;mask[:,:,:margin]=False;mask[:,:,-margin:]=False
    allids=mask.nonzero();assert len(allids)
    # Uniform samples plus boundary-neighbourhood samples prevent large flat
    # regions from making narrow objects vanish from the objective.
    dx=F.pad(torch.diff(target,dim=-1).abs().amax(1),(0,1));dy=F.pad(torch.diff(target,dim=-2).abs().amax(1),(0,0,0,1))
    boundary=F.max_pool2d(((dx+dy)>4/128).float()[:,None],9,1,4)[:,0].bool()&mask
    edges=boundary.nonzero()
    def pick(v,k):return v[torch.linspace(0,len(v)-1,min(k,len(v)),device=v.device).long()]
    return torch.cat((pick(allids,count//2),pick(edges,count-count//2))) if len(edges) else pick(allids,count)


def matching(p,t,count=384):
    assert p.shape==t.shape and min(p.shape[-2:])>=32
    ids=points(t.detach(),count);bb,yy,xx=ids.unbind(1);radius=4;shift=range(-8,9,2)
    dd=torch.arange(-radius,radius+1,device=p.device);dy,dx=torch.meshgrid(dd,dd,indexing='ij')
    def take(z,sy=0,sx=0):
        z=z.permute(0,2,3,1)
        if torch.is_tensor(sy):sy=sy[:,None,None];sx=sx[:,None,None]
        return z[bb[:,None,None],yy[:,None,None]+dy+sy,xx[:,None,None]+dx+sx]
    query_p,query_t=take(p),take(t)
    with torch.no_grad():
        # A target-based shared contrast floor keeps small valid layers visible
        # without giving almost-constant sensor variations unlimited gain.
        contrast=(query_t.amax((1,2))-query_t.amin((1,2))).amax(1).clamp_min(12/128)
        forward=torch.full((len(ids),),float('inf'),device=p.device);reverse=forward.clone()
        fy=torch.zeros_like(bb);fx=fy.clone();ry=fy.clone();rx=fy.clone()
        # Prefer zero/small displacement on exact ties.
        offsets=sorted(((a,b) for a in shift for b in shift),key=lambda v:v[0]**2+v[1]**2)
        for sy,sx in offsets:
            a=(query_p.detach()-take(t,sy,sx)).abs().mean((1,2,3))
            b=(query_t-take(p.detach(),sy,sx)).abs().mean((1,2,3))
            fa,rb=a<forward,b<reverse
            fy=torch.where(fa,sy,fy);fx=torch.where(fa,sx,fx);ry=torch.where(rb,sy,ry);rx=torch.where(rb,sx,rx)
            forward=torch.minimum(forward,a);reverse=torch.minimum(reverse,b)
    first=((query_p-take(t,fy,fx)).abs().mean((1,2,3))/contrast).mean()
    second=((query_t-take(p,ry,rx)).abs().mean((1,2,3))/contrast).mean()
    value=(first+second)/2
    return value,dict(patch_coherence=first,patch_coverage=second,patch_correspondence=value,
        patch_offset_mean=((fy.float().square()+fx.float().square()).sqrt().mean()+(ry.float().square()+rx.float().square()).sqrt().mean())/2)


def regression():
    t=torch.zeros(1,3,72,96);t[...,48:]=.5
    assert matching(t,t)[0]==0
    assert matching(t.roll(4,-1),t)[0]==0
    blur=F.avg_pool2d(F.pad(t,(4,4,4,4),mode='replicate'),9,1).requires_grad_()
    loss=matching(blur,t)[0];grad=torch.autograd.grad(loss,blur)[0]
    assert loss>.01 and torch.isfinite(grad).all() and (grad*(t-blur.detach())).sum()<0
    for width in (1,2,3,4):
        thin=t*0;thin[...,48:48+width]=.5
        assert matching(thin*0,thin)[0]>.01
    wrong=t.clone();wrong[:,0]=wrong[:,0].roll(6,-1);wrong[:,2]=wrong[:,2].roll(-6,-1)
    assert matching(wrong,t)[0]>.01
    corner=t*0;corner[...,36:,48:]=.5
    assert matching(corner,corner)[0]==0
    print('BIDIRECTIONAL_PATCH_PASS target/shift4/blur_gradient/thin1to4/jointRGB/corner',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
