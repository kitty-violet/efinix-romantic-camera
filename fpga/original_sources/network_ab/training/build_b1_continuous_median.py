"""Exact separable median-of-three before colour decisions, in existing B1.

Each coordinate uses ReLU min/max identities across the six residual PWs.
This keeps continuous coordinates until the existing360p stage, so removing
small texture does not require committing boundaries to180p binary pixels.
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
def build(net):
    for k,w in net.weight.items():w.zero_();net.bias[k].zero_()
    install_palette(net,LEVELS)
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['stem'][:3,:,1,1]=net.weight['stem'].new_tensor([[1,2,1],[2,0,-2],[-1,2,-1]])/8;net.bias['stem'][:3]=64
    net.weight['down1__dw'][:3,0]=kernel
    for g in range(3):net.weight['down1__pw'][g,g,0,0]=1
    net.weight['down2__dw'][:3,0]=kernel
    for g in range(3):
        for j in range(3):net.weight['down2__pw'][4*g+j,g,0,0]=1
    for k in ('res1_conv2__dw','res2_conv1__dw','res3_conv1__dw','res3_conv2__dw'):net.weight[k][:12,0,1,1]=1
    for g in range(3):
        for j in range(3):net.weight['res1_conv1__dw'][4*g+j,0,1,j]=1;net.weight['res2_conv2__dw'][4*g+j,0,j,1]=1
    # First comparison: [ReLU(a-b),a,b,c].
    for k in ('res1_conv1__pw','res2_conv2__pw'):
        for g in range(3):
            o=4*g;net.weight[k][o,o,0,0]=1;net.weight[k][o,o+1,0,0]=-1
            for dst,src in ((1,0),(2,1),(3,2)):net.weight[k][o+dst,o+src,0,0]=1
    # [min(a,b)-c, c-max(a,b), c]; the next ReLU clips the two differences.
    for k in ('res1_conv2__pw','res3_conv1__pw'):
        for g in range(3):
            o=4*g
            for src,v in ((o+1,1),(o,-1),(o+3,-1)):net.weight[k][o,src,0,0]=v
            for src,v in ((o+3,1),(o,-1),(o+2,-1)):net.weight[k][o+1,src,0,0]=v
            net.weight[k][o+2,o+3,0,0]=1
    for g in range(3):
        o=4*g
        for dst in (o,o+1,o+2):
            for src,v in ((o+2,1),(o,1),(o+1,-1)):net.weight['res2_conv1__pw'][dst,src,0,0]=v
        for src,v in ((o+2,1),(o,1),(o+1,-1)):net.weight['res3_conv2__pw'][g,src,0,0]=v
    net.weight['up1_low_dw'][:3,0,1,1]=1;net.weight['up1_low_pw'].zero_();net.bias['up1_low_pw'].zero_()
    for g,(start,end) in enumerate(GROUPS):
        v=np.array(LEVELS[g],float);cut=(v[:-1]+v[1:])/2
        if g==0:raw=(cut+8)/1.06/2
        elif g==1:raw=(cut+8)/1.20/2+64
        else:raw=cut/1.15/2+64
        for j,t in enumerate(np.ceil(raw).astype(int)):
            net.weight['up1_low_pw'][start+j,g,0,0]=1;net.bias['up1_low_pw'][start+j]=64-int(t)
    for name in ('res1_add','res2_add','res3_add'):net.package['residual_encoding'][name]=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for k,w in net.weight.items():
        if k in ('up2_low_pw','up2_low_dw'):continue
        s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    net=m.t.r.initialize();net.quantized=True;build(net)
    field=torch.randint(0,128,(16,3,16,16),device='cuda').float();z=torch.zeros(16,32,16,16,device='cuda')
    for g in range(3):z[:,4*g:4*g+3]=field[:,g:g+1]
    part=fragment(net,z,'down2.pw');horizontal=F.pad(field,(1,1,0,0),mode='replicate').unfold(-1,3,1).median(-1).values
    expected=F.pad(horizontal,(0,0,1,1),mode='replicate').unfold(-2,3,1).median(-1).values
    error=float((part['res3_add'][:,:3]-expected).abs().max());assert error==0,error
    d=m.load_data();export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        ordered_regions=True,manual_continuous_median=True,coverage_layout='15ordered360_rgb200',trained=False),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(separable_median_values_checked=16*3*16*16,separable_median_error=error,
        manually_designed=True,trained=False,decision_resolution='640x360',added_operators=0,postfilter=False,
        global_minimum_area_guaranteed=False,visual_acceptance=False))
    print('CONTINUOUS_MEDIAN_PACKAGE_PASS',error,flush=True)


if __name__=='__main__':main()
