"""Binary level-set majority as existing B1 coefficients, no new operators.

Sixteen region planes and sixteen copies let each residual replace a plane by
its 3x3 majority: skip + (majority - original). Colours are recombined only
after three spatial majority stages. This is an explicit initialization.
"""
import math
import torch
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .runtime import configure_workspace,write_json
from .engine import set_reproducible


@torch.no_grad()
def build():
    net=t.r.initialize();net.quantized=True
    for key in net.weight:net.weight[key].zero_();net.bias[key].zero_()
    def dw(key,channels=None,blur=False):
        w=net.weight[key];channels=range(len(w)) if channels is None else channels
        kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
        for c in channels:
            if blur:w[c,0]=kernel
            else:w[c,0,1,1]=1
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    dw('down1__dw',range(3),True)
    matrix=net.weight['stem'].new_tensor([[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]])
    for j in range(16):
        net.weight['down1__pw'][j,:3,0,0]=matrix[j%3]
        net.bias['down1__pw'][j]=0 if j%3==0 else 64
    dw('down2__dw',blur=True)
    # Chroma must have a neutral central interval. Thresholding at neutral64
    # created alternating green/purple areas on nominally grey surfaces.
    thresholds=[(0,v) for v in [14,28,42,56,70,84,98,112]]+[(1,v) for v in [49,59,69,79]]+[(2,v) for v in [49,59,69,79]]
    for j,(channel,threshold) in enumerate(thresholds):
        for c in (j,j+16):
            net.weight['down2__pw'][c,channel,0,0]=16
            net.bias['down2__pw'][c]=64-16*threshold
    for k in range(1,4):
        a=f'res{k}_conv1__dw';dw(a,range(16),True);dw(a,range(16,32))
        a=f'res{k}_conv1__pw'
        for c in range(16):
            net.weight[a][c,c,0,0]=16;net.bias[a][c]=64-16*63.5
            net.weight[a][c+16,c+16,0,0]=1
        dw(f'res{k}_conv2__dw')
        a=f'res{k}_conv2__pw'
        for c in range(16):
            for j in (c,c+16):
                net.weight[a][j,c,0,0]=1;net.weight[a][j,c+16,0,0]=-1
    dw('up1_low_dw')
    for c in range(16):net.weight['up1_low_pw'][c,c,0,0]=1
    dw('up1_high_dw',blur=True)
    # Geometric interpolation of level sets, then re-steepen before colour mix.
    for c in range(16):
        net.weight['up2_low_dw'][c,0,1,1]=2;net.bias['up2_low_dw'][c]=-63.5
    con,sat=1.04,1.08
    for c in range(3):
        net.weight['up2_low_pw'][c,:8,0,0]=14*con/127
        if c==0:
            net.weight['up2_low_pw'][c,8:12,0,0]=10*sat/127
            net.weight['up2_low_pw'][c,12:,0,0]=-10*sat/127
            net.bias['up2_low_pw'][c]=7*con+64*(1-con)-1
        elif c==1:
            net.weight['up2_low_pw'][c,12:,0,0]=10*sat/127
            net.bias['up2_low_pw'][c]=7*con+(44-64)*sat+64*(1-con)
        else:
            net.weight['up2_low_pw'][c,8:12,0,0]=-10*sat/127
            net.weight['up2_low_pw'][c,12:,0,0]=-10*sat/127
            net.bias['up2_low_pw'][c]=7*con-(44+44-128)*sat+64*(1-con)+1.5
    dw('up2_high_dw',range(3),True)
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2;net.bias['rgb_head'][c]=-128
    # Power-of-two scales represent the integer/quarter/binomial majority
    # arithmetic exactly; arbitrary trained scales are not silently reused.
    for key,w in net.weight.items():
        peak=w.abs().flatten(1).amax(1).clamp_min(1e-5)
        scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
        multiplier=(scale*131072).clamp(1,131071)
        getattr(net,key+'_m').copy_(multiplier.double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100637)
    dest=OUT/'binary-region-neutral';dest.mkdir(exist_ok=False)
    net=build();raw=net.export();(dest/'initial.bin').write_bytes(raw)
    actual=t.r.IntegerB1(t.r.runtime_from_image(raw));d=t.data();errors=[];states=[]
    for row in d['validation']:
        x=t.r.tensor(t.r.read(row['path']))[None];ours=net(x,all_nodes=True);ref,_=actual(x*128,True)
        errors.append(max(float((v-ref[k]).abs().max()) for k,v in ours.items()))
        states.append({key:sorted(float(v) for v in ours[key].unique()) for key in ('res1_add','res2_add','res3_add')})
    assert max(errors)==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,parameter_initialization_only=True),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview')
    write_json(dest/'review.json',dict(all_node_error=max(errors),states=states,minimum_area_guarantee=False,
        training_steps=0,added_operators=0,description='16 cumulative soft-clipped level sets, three 3x3 weighted majority stages, neutral-centred chroma; not exact binary states'))
    print('BINARY_REGION_INIT',max(errors),states,flush=True)


if __name__=='__main__':main()
