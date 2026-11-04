"""Manually construct ordered region cleanup entirely inside the fixed B1.

RGB->YCoCg->ordinal regions->binary majority filters->palette RGB. Majority
filters are median filters for the ordered colour coordinates. This does not
assert a universal component-area lower bound, and never adds an output filter.
"""
import argparse
import copy
import shutil
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import GROUPS,actual_palette,install as install_palette
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json

LEVELS=[[20,48,76,104,132,160,192,228],[-56,-32,-8,16,40],[-40,-20,0,20,40]]


@torch.no_grad()
def build(net,passes,single_aa=False):
    for k,w in net.weight.items():w.zero_();net.bias[k].zero_()
    install_palette(net,LEVELS)
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['stem'][:3,:,1,1]=net.weight['stem'].new_tensor([[1,2,1],[2,0,-2],[-1,2,-1]])/8
    net.bias['stem'][:3]=64
    net.weight['down1__dw'][:3,0]=kernel
    # Source thresholds compensate the fixed contrast/saturation/cool tone.
    for g,(start,end) in enumerate(GROUPS):
        levels=np.array(LEVELS[g],float);cut=(levels[:-1]+levels[1:])/2
        if g==0:raw=(cut+8)/1.06/2
        elif g==1:raw=(cut+8)/1.20/2+64
        else:raw=cut/1.15/2+64
        threshold=np.ceil(raw).astype(int)
        for j,t in enumerate(threshold):
            net.weight['down1__pw'][start+j,g,0,0]=127
            net.bias['down1__pw'][start+j]=127*(1-int(t))
    # Binary input0/127: sum>=5*127 produces127; smaller sums produce0.
    net.weight['down2__dw'][:15,0]=1;net.bias['down2__dw'][:15]=-4*127
    for j in range(15):net.weight['down2__pw'][j,j,0,0]=1
    layers=['res1_conv1__dw','res1_conv2__dw','res2_conv1__dw','res2_conv2__dw','res3_conv1__dw','res3_conv2__dw']
    active=set(layers[:passes])
    for k in layers:
        if k in active:net.weight[k][:15,0]=1;net.bias[k][:15]=-4*127
        else:net.weight[k][:15,0,1,1]=1
    for k in ('res1_conv1__pw','res1_conv2__pw','res2_conv1__pw','res2_conv2__pw','res3_conv1__pw','res3_conv2__pw','up1_low_pw'):
        net.weight[k].zero_();net.bias[k].zero_()
        for j in range(15):net.weight[k][j,j,0,0]=1
    net.weight['up1_low_dw'][:15,0,1,1]=1
    net.weight['up1_high_dw'].zero_();net.weight['up1_high_dw'][:15,0]=1;net.bias['up1_high_dw'][:15]=-4*127
    # Interpolate the cumulative indicators, then do only linear colour mixing.
    net.weight['up2_low_dw'].zero_();net.bias['up2_low_dw'].zero_()
    if single_aa:net.weight['up2_low_dw'][:15,0,1,1]=1
    else:net.weight['up2_low_dw'][:15,0]=kernel
    for name in ('res1_add','res2_add','res3_add'):
        net.package['residual_encoding'][name]=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for k,w in net.weight.items():
        if k=='up2_low_pw':continue # Nonuniform palette increments already calibrated.
        maximum=w.abs().flatten(1).amax(1)
        s=torch.pow(2,torch.ceil(torch.log2(maximum.clamp_min(1e-5)/127)))
        if k=='down1__pw':s[:15]=131071/131072
        mult=(s*131072).clamp(1,131071)
        getattr(net,k+'_m').copy_(mult.double()[None,:,None,None])
        if k=='down1__pw':
            net.weight[k][:15]*=131071/131072;net.bias[k][:15]*=131071/131072


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--single-aa',action='store_true');a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);d=m.load_data()
    for passes in (0,2,4,6):
        net=m.t.r.initialize();net.quantized=True;build(net,passes,a.single_aa);path=dest/f'passes{passes}';path.mkdir()
        # All stages through the final region filter must remain ordered and binary.
        rows=[]
        for row in d['validation']:
            nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True);z=nodes['up1_high_dw'][:,:15]
            assert torch.all((z==0)|(z==127))
            for start,end in GROUPS:assert torch.all(z[:,start:end-1]>=z[:,start+1:end])
            rows.append(row['id'])
        export(net,d['validation'],path/'package');m.t.preview(net,d['validation'],path/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            manual_median_regions=True,median_passes180=passes,single_aa=a.single_aa,coverage_layout='15ordered_median_rgb200',trained=False),path/'initial.pt')
        print('MANUAL_MEDIAN_PACKAGE',passes,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(passes180=[0,2,4,6],source_thresholds=dict(contrast=1.06,brightness=-8,saturationCo=1.2,saturationCg=1.15,coolCo=-8),
        palette_levels=LEVELS,single_aa=a.single_aa,manually_designed=True,trained=False,added_operators=0,postfilter=False,global_minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
