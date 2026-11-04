"""Train the existing360 depthwise boundary kernel, with floating unity DC.

Only existing coefficients change. The float kernel is nonnegative with sum1;
the rounded integer sum must be audited separately, not assumed to equal1.
"""
import torch


@torch.no_grad()
def project_kernel(net):
    w=net.weight['up1_high_dw']
    scale=(net.up1_high_dw_m/131072).float().flatten()
    cap=127*scale[:15,None]
    rows=w[:15,0].flatten(1)
    # Euclidean projection onto the nonnegative capped unit simplex.
    lo=(rows-cap).amin(1,keepdim=True)-1;hi=rows.amax(1,keepdim=True)+1
    for _ in range(28):
        mid=(lo+hi)/2;value=(rows-mid).clamp_min(0).minimum(cap)
        large=value.sum(1,keepdim=True)>1
        lo=torch.where(large,mid,lo);hi=torch.where(large,hi,mid)
    rows.copy_((rows-(lo+hi)/2).clamp_min(0).minimum(cap))
    w[15:].zero_();net.bias['up1_high_dw'].zero_()
    assert (rows.sum(1)-1).abs().max()<1e-5 and rows.min()>=0


def enable_kernel(net,opt,lr):
    key='up1_high_dw';w=net.weight[key];w.requires_grad_(True)
    net.bias[key].requires_grad_(False)
    s=(getattr(net,key+'_m')/131072).float().flatten()
    opt.add_param_group(dict(params=[w],lr=lr,row_gain=(s/(1/256))[:,None,None,None]))
    project_kernel(net)
