"""Finer existing Q17 coefficients and live decoder channels, same B1 graph.

Exact re-encoding of the initial colour path is tested, not assumed. New
channels initially have zero head contribution. All later outputs are actual
learned coefficients; no filter is appended to inference.
"""
import torch


@torch.no_grad()
def install(net):
    assert net.weight['rgb_head'][:,3:].count_nonzero()==0
    assert torch.all(net.up1_high_dw_m==8738)
    net.up1_high_dw_m.div_(17)
    net.up2_low_dw_m[:,:14].fill_(16384)
    net.up2_low_dw_m[:,14:].fill_(2048)
    # This checkpoint has8257, not8256. Keep its live Y row exactly: replacing
    # it with64*129 is not globally bit-identical on fractional decisions.
    assert net.up2_low_pw_m[0,0,0,0] in (8256,8257)
    net.up2_low_pw_m[:,1:3].fill_(2048)
    for c in range(3,6):
        src=c-3
        net.weight['up2_low_pw'][c].copy_(net.weight['up2_low_pw'][src])
        net.bias['up2_low_pw'][c].copy_(net.bias['up2_low_pw'][src])
        net.up2_low_pw_m[:,c].copy_(net.up2_low_pw_m[:,src])
    # The initially silent extra Y channel can use fine precision safely.
    net.up2_low_pw_m[:,3].fill_(129)
    for c,src in ((6,4),(7,10)):
        net.weight['up2_low_pw'][c].zero_();net.weight['up2_low_pw'][c,src,0,0]=1
        net.bias['up2_low_pw'][c].zero_();net.up2_low_pw_m[:,c].fill_(2048)
    kernel=net.weight['up2_high_dw'][0,0].clone()
    net.weight['up2_high_dw'][3:,0].copy_(kernel);net.bias['up2_high_dw'][3:].zero_()
    net.up2_high_dw_m.fill_(512);net.rgb_head_m.fill_(4096)
    for k,w in net.weight.items():
        scale=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert net.round(w/scale).abs().max()<=127,('Unrepresentable row',k)
