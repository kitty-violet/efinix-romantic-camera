"""Isolate360 support width after moving the final count to720.

Existing wide360 vote and720 interpolation can spread one boundary twice.
Replace only the360 kernel with a point sample; keep its constant response.
This is a diagnostic and is not an acceptance claim.
"""
import torch
from . import probe_b1_ternary_packed720 as probe


original_install=probe.install


@torch.no_grad()
def install(net):
    setup=original_install(net);key='up2_low_dw'
    s=(getattr(net,key+'_m')/131072).float().flatten()
    effective=net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
    dc=effective.sum((1,2,3));net.weight[key].zero_();net.weight[key][:,0,1,1]=dc
    mult=(dc.abs()*131072/126).ceil().clamp(1,131071);getattr(net,key+'_m').copy_(mult.double()[None,:,None,None])
    setup['vote360_support']='center_preserved_integer_DC';return setup


if __name__=='__main__':
    probe.install=install
    probe.main()
