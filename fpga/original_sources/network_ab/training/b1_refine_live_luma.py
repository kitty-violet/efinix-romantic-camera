"""Exact divisor re-encoding of a live INT8/Q17 coefficient row."""
import torch


@torch.no_grad()
def refine(net):
    key='up2_low_pw';c=0;oldm=int(getattr(net,key+'_m')[0,c,0,0]);assert oldm==8257
    factor=23;assert oldm%factor==0
    scale=oldm/131072;w=net.round(net.weight[key][c]/scale);b=net.round(net.bias[key][c]/scale)
    assert w.abs().max()*factor<=127 and b.abs()*factor<2**31
    # Save the exact currently deployed coefficients, not fractional masters.
    net.weight[key][c].copy_(w*scale);net.bias[key][c].copy_(b*scale)
    getattr(net,key+'_m')[:,c].fill_(oldm//factor)
    assert torch.equal(net.round(net.weight[key][c]/((oldm//factor)/131072)),w*factor)
    assert torch.equal(net.round(net.bias[key][c]/((oldm//factor)/131072)),b*factor)
    return dict(old_multiplier=oldm,new_multiplier=oldm//factor,precision_factor=factor)
