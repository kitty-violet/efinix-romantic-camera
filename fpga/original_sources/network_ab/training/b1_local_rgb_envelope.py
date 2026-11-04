"""Target-local RGB-line gamut guard plus interior error variance.

At approximately two-colour boundaries, true mixtures lie near the same RGB
line segment. Orthogonal colours and endpoint overshoots are separately
penalised; broadening an edge cannot substitute for the geometry objectives.
Multi-colour junctions are excluded from the rank-one guard. Training only.
"""
import torch
from torch.nn import functional as F


def local_envelope(p,t,labels,maximum=256):
    seam=torch.zeros_like(labels,dtype=torch.bool)
    seam[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    seam[:,1:]|=labels[:,1:]!=labels[:,:-1]
    seam[:,:,:6]=False;seam[:,:,-6:]=False;seam[:,:6]=False;seam[:,-6:]=False
    idx=seam.nonzero()
    if len(idx)>maximum:idx=idx[torch.linspace(0,len(idx)-1,maximum,device=p.device).long()]
    if not len(idx):
        zero=p.sum()*0
        return zero,dict(rgb_line_fringe=zero,rgb_line_overshoot=zero,rgb_line_samples=zero.detach())
    bb,yy,xx=idx.unbind(1);d=torch.arange(-4,5,device=p.device);dy,dx=torch.meshgrid(d,d,indexing='ij')
    def gather(z):return z.permute(0,2,3,1)[bb[:,None,None],yy[:,None,None]+dy,xx[:,None,None]+dx].flatten(1,2)
    reference=gather(t).detach();q=gather(p)
    centre=reference.mean(1,keepdim=True);r=reference-centre
    cov=r.transpose(-1,-2)@r/r.shape[1]
    eig,vec=torch.linalg.eigh(cov);v=vec[:,:,-1]
    select=(eig[:,-1]>(8/128)**2)&(eig[:,-2]<.02*eig[:,-1])
    rpos=(r*v[:,None]).sum(-1);lo=rpos.amin(1);hi=rpos.amax(1);span=(hi-lo).clamp_min(16/128)
    # Fixed allowance comes from the target residual and the1--2 byte codec
    # scale; the student is not allowed to expand its own permitted gamut.
    rperp=(r-rpos[...,None]*v[:,None]).norm(dim=-1)
    tolerance=rperp.amax(1)+2/128
    delta=q-centre;pos=(delta*v[:,None]).sum(-1)
    perp=(delta-pos[...,None]*v[:,None]).norm(dim=-1)
    colour=(F.relu(perp-tolerance[:,None])/span[:,None]).mean(1)
    over=((F.relu(lo[:,None]-pos-2/128)+F.relu(pos-hi[:,None]-2/128))/span[:,None]).mean(1)
    if select.any():colour=colour[select].mean();over=over[select].mean()
    else:colour=p.sum()*0;over=colour
    return colour+over,dict(rgb_line_fringe=colour,rgb_line_overshoot=over,rgb_line_samples=select.sum().to(p.dtype))


def core_error_variation(p,t,labels):
    seam=torch.zeros_like(labels,dtype=torch.bool)
    seam[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    seam[:,1:]|=labels[:,1:]!=labels[:,:-1]
    core=(F.max_pool2d(seam[:,None].float(),7,1,3)==0)[:,0]
    values=[]
    for err,lab,mask in zip(p-t,labels,core):
        if not mask.any():continue
        _,index=torch.unique(lab[mask],return_inverse=True);count=torch.bincount(index).to(p.dtype)
        e=err[:,mask]
        # Large constant offsets must not acquire spurious variance through
        # repeated float32 summation. Accumulate regional means in float64.
        mean=e.double().new_zeros(3,len(count)).scatter_add(1,index[None].expand_as(e),e.double())/count.double()
        mean=mean.to(e.dtype)
        diff=(e-mean[:,index]).abs().mean(0)
        per=diff.new_zeros(len(count)).scatter_add(0,index,diff)/count
        weights=count.sqrt();values.append((per*weights).sum()/weights.sum())
    return torch.stack(values).mean() if values else p.sum()*0


def regression():
    lab=torch.zeros(1,64,96,dtype=torch.long);lab[...,48:]=1
    a=torch.tensor([-.5,-.3,.1])[None,:,None,None];d=torch.tensor([.4,.6,.2])[None,:,None,None]
    target=a+lab[:,None]*d
    assert local_envelope(target,target,lab)[0]==0
    # Any common RGB mixture between both endpoint colours is legal. This
    # guard alone does not reward sharpness and must never replace shape loss.
    blurred=F.avg_pool2d(F.pad(target,(2,2,2,2),mode='replicate'),5,1)
    assert local_envelope(blurred,target,lab)[0]<1e-6
    wrong=target.clone();wrong[...,46:50]+=torch.tensor([.2,-.2,.2])[None,:,None,None]
    assert local_envelope(wrong,target,lab)[1]['rgb_line_fringe']>.01
    overshoot=target.clone();overshoot[...,48:51]+=d
    assert local_envelope(overshoot,target,lab)[1]['rgb_line_overshoot']>.01
    wrong.requires_grad_();loss,_=local_envelope(wrong,target,lab);loss.backward()
    assert torch.isfinite(wrong.grad).all() and (wrong.grad*(target-wrong.detach())).sum()<0
    assert core_error_variation(target,target,lab)==0
    assert core_error_variation(target+.1,target,lab)<1e-6
    noise=target.clone();noise[...,::2,::2]+=.1
    assert core_error_variation(noise,target,lab)>.01
    print('LOCAL_RGB_ENVELOPE_PASS exact/convex_mix/false_colour/overshoot/finite_gradient/core_noise',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
