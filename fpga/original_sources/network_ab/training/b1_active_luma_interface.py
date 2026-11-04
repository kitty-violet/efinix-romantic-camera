"""Align only Y predicates that name the same active target interface.

Historical coincidence averaged saturated interiors with real interfaces.
Here all participating target fields must lie inside the distance transition
and share the same value (within one-quarter code); distinct edges stay free.
The field-space gradients sum to zero within each participating group.
"""
import torch
from torch.nn import functional as F


def active_interface_loss(field,truth):
    assert field.shape==truth.shape and field.shape[1]==14
    split=torch.diff(truth,dim=1).abs()>.25
    ids=F.pad(split.long().cumsum(1),(0,0,0,0,1,0))
    p=field/127;t=truth/127
    count=torch.zeros_like(t).scatter_add(1,ids,torch.ones_like(t)).clamp_min(1)
    pm=(torch.zeros_like(p).scatter_add(1,ids,p)/count).gather(1,ids)
    tm=(torch.zeros_like(t).scatter_add(1,ids,t)/count).gather(1,ids)
    # Exclude both constant interior groups; only the shared boundary remains.
    mask=(count.gather(1,ids)>=2)&(truth>4)&(truth<123)
    error=F.smooth_l1_loss(p-pm,t-tm,beta=.05,reduction='none')
    return (error*mask).sum()/mask.sum().clamp_min(1),mask.float().mean()


def regression():
    t=torch.zeros(1,14,12,24);t[:,:3]=127;t[:,3:8]=(63.5+8*(torch.arange(24)-11.5)).clamp(0,127)
    assert active_interface_loss(t,t)[0]==0
    common=t.clone();common[:,3:8]=(common[:,3:8]+10).clamp(0,127)
    # On the active target band, a common displacement changes no relative
    # edge position; target-field/decision objectives still learn the mean.
    assert active_interface_loss(common,t)[0].abs()<1e-12
    p=t.clone();p[:,4:6]=(p[:,4:6]+20).clamp(0,127);p.requires_grad_()
    v,f=active_interface_loss(p,t);v.backward();assert v>0 and 0<f<1
    assert p.grad[:,3:8].sum(1).abs().max()<1e-7 and torch.isfinite(p.grad).all()
    assert p.grad[:,:3].abs().max()==0 and p.grad[:,8:].abs().max()==0
    assert active_interface_loss(torch.zeros_like(t),torch.zeros_like(t))[0]==0
    print('ACTIVE_LUMA_INTERFACE_PASS target0/common_shift0/split_gradient_sum0/interiors_excluded/empty0',flush=True)


if __name__=='__main__':regression()
