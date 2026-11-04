"""Revive five existing 720p decoder channels as zero-DC spatial features.

No graph changes. The old colour path is initially bit-identical. The five
features have constant response64; compensate their quantized RGB-head DC
exactly in its existing bias. Output correction is bounded by32 byte codes.
This is a capacity experiment, not an accepted sharpening algorithm.
"""
import copy
import torch


@torch.no_grad()
def install(net):
    before=copy.deepcopy(net.state_dict())
    assert net.weight['rgb_head'][:,3:].count_nonzero()==0
    assert net.weight['up2_high_dw'][3:].count_nonzero()==0
    # Finer representable head quantum; old2,1.125,.75 remain exact.
    net.rgb_head_m.fill_(4096)
    for c in range(3,6):
        src=c-3
        net.weight['up2_low_pw'][c].copy_(net.weight['up2_low_pw'][src])
        net.bias['up2_low_pw'][c].copy_(net.bias['up2_low_pw'][src])
        net.up2_low_pw_m[:,c].copy_(net.up2_low_pw_m[:,src])
    for c,lo,hi in ((6,2,8),(7,8,13)):
        net.weight['up2_low_pw'][c].zero_()
        net.weight['up2_low_pw'][c,lo,0,0]=.5
        net.weight['up2_low_pw'][c,hi,0,0]=-.5
        net.bias['up2_low_pw'][c].fill_(32)
        net.up2_low_pw_m[:,c].fill_(2048)
    kernel=torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],device=net.rgb_head_m.device)/8
    net.weight['up2_high_dw'][3:,0].copy_(kernel)
    net.bias['up2_high_dw'][3:].fill_(64)
    net.up2_high_dw_m[:,3:].fill_(16384)
    return before


@torch.no_grad()
def compensate_head(net,base_bias):
    # Integer head weights and bias share one1/32 quantum; exact cancellation.
    s=(net.rgb_head_m/131072).float().flatten()
    w=net.weight['rgb_head'][:,3:]
    w.clamp_(-.5,.5)
    # Project integer coefficients, so the bound is about deployed weights.
    q=net.round(w/s[:,None,None,None])
    # Sixteen integer units means sum(abs(effective coefficient))<=.5.
    for row in range(3):
        changed=False
        while int(q[row].abs().sum())>16:
            changed=True
            index=int(q[row].abs().flatten().argmax())
            q[row].view(-1)[index]-=q[row].view(-1)[index].sign()
        if changed:w[row].copy_(q[row]*s[row])
    net.bias['rgb_head'].copy_(base_bias-64*(q*s[:,None,None,None]).sum((1,2,3)))


def masks(net):
    result={}
    for key,parameter in net.weight.items():
        if key=='up2_low_pw':
            mask=torch.zeros_like(parameter);mask[3:]=1;result[key]=mask
        elif key=='rgb_head':
            mask=torch.zeros_like(parameter);mask[:,3:]=1;result[key]=mask
    return result
