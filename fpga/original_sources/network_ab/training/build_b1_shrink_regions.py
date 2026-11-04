"""Three bounded, edge-preserving shrink steps compiled into existing B1.

For each YCoCg coordinate: u <- mean(u) + soft_threshold(u-mean(u), t).
The correction is bounded by t. This is a local, finite shrink scheme, NOT
the global L0 solver and NOT a guaranteed connected-component area filter.
Each pair of existing PWs evaluates one step using two ReLU differences.
Colour decisions are delayed to360p; the transition width is configurable.
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import GROUPS,install as install_palette
from .build_b1_median_cartoon import LEVELS
from .probe_b1_decoder_coverage import fragment
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def build(net,threshold,transition):
    for k,w in net.weight.items():w.zero_();net.bias[k].zero_()
    install_palette(net,LEVELS)
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['stem'][:3,:,1,1]=net.weight['stem'].new_tensor([[1,2,1],[2,0,-2],[-1,2,-1]])/8
    net.bias['stem'][:3]=64
    for g in range(3):
        net.weight['down1__dw'][g,0]=kernel
        net.weight['down1__pw'][g,g,0,0]=1
        net.weight['down2__dw'][g,0]=kernel
        for j in range(2):net.weight['down2__pw'][2*g+j,g,0,0]=1
    for block in range(1,4):
        dw1=f'res{block}_conv1__dw';pw1=f'res{block}_conv1__pw'
        dw2=f'res{block}_conv2__dw';pw2=f'res{block}_conv2__pw'
        for g in range(3):
            a=2*g;b=3*g
            net.weight[dw1][a,0]=kernel;net.weight[dw1][a+1,0,1,1]=1
            net.weight[pw1][b,a,0,0]=1
            net.weight[pw1][b+1,a+1,0,0]=1;net.weight[pw1][b+1,a,0,0]=-1
            net.weight[pw1][b+2,a,0,0]=1;net.weight[pw1][b+2,a+1,0,0]=-1
            net.bias[pw1][b+1:b+3]=-threshold
            net.weight[dw2][b:b+3,0,1,1]=1
            for j in range(2):
                for c,v in ((b,1),(b+1,1),(b+2,-1)):net.weight[pw2][a+j,c,0,0]=v
        net.package['residual_encoding'][f'res{block}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    net.weight['up1_low_dw'][:6,0,1,1]=1
    net.weight['up1_low_pw'].zero_();net.bias['up1_low_pw'].zero_()
    for g,(start,end) in enumerate(GROUPS):
        v=np.array(LEVELS[g],float);cut=(v[:-1]+v[1:])/2
        if g==0:raw=(cut+8)/1.06/2
        elif g==1:raw=(cut+8)/1.20/2+64
        else:raw=cut/1.15/2+64
        for j,t in enumerate(np.ceil(raw).astype(int)):
            net.weight['up1_low_pw'][start+j,2*g,0,0]=1
            net.bias['up1_low_pw'][start+j]=64-int(t)
    # Narrow transition affects only colour-boundary coverage, not every pixel.
    net.weight['up2_low_dw'].zero_();net.bias['up2_low_dw'].zero_()
    gain=127/transition;lo=64-(transition+1)/2
    net.weight['up2_low_dw'][:15,0,1,1]=gain
    net.bias['up2_low_dw'][:15]=-lo*gain
    for k,w in net.weight.items():
        if k=='up2_low_pw':continue
        s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def verify(net,threshold):
    gen=torch.Generator(device='cuda').manual_seed(19210)
    u=torch.randint(0,128,(16,3,16,16),generator=gen,device='cuda').float()
    z=torch.zeros(16,32,16,16,device='cuda')
    for g in range(3):z[:,2*g:2*g+2]=u[:,g:g+1]
    part=fragment(net,z,'down2.pw');expected=u
    kernel=u.new_tensor([[1,2,1],[2,4,2],[1,2,1]])[None,None].expand(3,1,3,3)
    for i in range(3):
        # Project round-to-nearest, ties away from zero, for nonnegative values.
        mean=torch.floor(F.conv2d(F.pad(expected,(1,1,1,1),mode='replicate'),kernel,groups=3)/16+.5)
        d=expected-mean;expected=mean+F.relu(d-threshold)-F.relu(-d-threshold)
        actual=part[f'res{i+1}_add'][:,[0,2,4]]
        error=float((actual-expected).abs().max());assert error==0,(i,error)
    return dict(random_coordinate_values=12288,iterations=3,error=error,max_correction_per_iteration=threshold)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);d=m.load_data()
    records=[]
    for threshold,transition in ((4,1),(8,1),(12,1),(4,4),(8,4),(12,4)):
        net=m.t.r.initialize();net.quantized=True;build(net,threshold,transition)
        check=verify(net,threshold);sub=dest/f't{threshold}_r{transition}';sub.mkdir()
        # Numeric export checks use the actual complete graph and independent oracle.
        export(net,d['validation'],sub/'package');m.t.preview(net,d['validation'],sub/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            manual_shrink_regions=True,ordered_regions=True,threshold=threshold,transition=transition,
            coverage_layout='15ordered360_rgb200',trained=False),sub/'initial.pt')
        records.append(dict(threshold=threshold,transition=transition,check=check))
        print('SHRINK_PACKAGE',threshold,transition,check,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,manually_designed=True,trained=False,added_operators=0,
        postfilter=False,global_minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
