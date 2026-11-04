"""Spend capacity at low resolution, retaining the checked 8-channel output.

Same 27-node operator graph; double only the formerly 16-channel context.
This requires new coefficients/descriptors; no production source is changed.
"""
import argparse,copy,sys
from pathlib import Path
import torch
from torch import nn
from .coarse_skip_net import CoarseSkipNet,plan as previous_plan
from .probe_boundary_budget import HW,tile
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json


def plan():
    p=copy.deepcopy(previous_plan());channels={'input':3}
    for n in p['nodes']:
        ci=channels[n['inputs'][0]];co=32 if n['channels']==16 else n['channels'];channels[n['id']]=co
        n.update(input_channels=ci,channels=co);n['input_shape_hwc'][2]=ci;n['output_shape_hwc'][2]=co
        if n['kernel']:
            n['groups']=ci if n['kind']=='dw3x3' else 1;n['terms']=ci*n['kernel']**2//n['groups']
            h,w,_=n['output_shape_hwc'];n['macs']=h*w*co*n['terms']
    p.update(macs=sum(n['macs'] for n in p['nodes']),production=False);return p


def fused_plan():
    nodes=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=nodes.pop();assert p['id']==n['inputs'][0] and p['kind']=='pw1x1'
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        nodes.append(n)
    nodes[-1]['id']='rgb_head';return dict(nodes=nodes)


class Surface32Net(CoarseSkipNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes']
        for n in self.nodes:
            if not n['kernel']:continue
            c=self.conv(n['id'])
            if c.in_channels==n['input_channels'] and c.out_channels==n['channels']:continue
            c=nn.Conv2d(n['input_channels'],n['channels'],n['kernel'],n['stride'],groups=n['groups'])
            parts=([nn.ReplicationPad2d(1)] if n['kernel']==3 else [])+[c]
            if n['activation']=='relu':parts.append(nn.ReLU())
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)

    @torch.no_grad()
    def expand_from(self,old):
        for n in self.nodes:
            if not n['kernel']:continue
            dst=self.conv(n['id']);src=old.conv(n['id']);oc=torch.arange(dst.out_channels)%src.out_channels
            if n['kind']=='dw3x3':w=src.weight[oc]
            else:
                ic=torch.arange(dst.in_channels)%src.in_channels;w=src.weight[oc][:,ic]/(dst.in_channels/src.in_channels)
            dst.weight.copy_(w);dst.bias.copy_(src.bias[oc])

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='surface32-luma-int8-q17-v1');return p


def main():
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=32);p.add_argument('--name',default='surface32-budget32-v1');a=p.parse_args()
    torch.set_num_threads(2);out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);sys.path.insert(0,str(HW))
    from fourway_frame_program import compile_tile
    g=fused_plan();rows=[]
    for i,x in enumerate(range(0,1280,a.width)):
        try:r=compile_tile(g,dict(name='research_surface32'),tile(g,i,x,min(a.width,1280-x)),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        except Exception as e:write_json(out/'audit.json',dict(status='ALLOCATION_FAIL',x=x,width=a.width,error=str(e)));raise
        rows.append(dict(x=x,high=r['finite_high_water']));print('SURFACE32_BUDGET',i+1,flush=True)
    weight=1792;bias=12608;mult=15040;relu=17472
    for n in plan()['nodes']:
        if not n['kernel']:continue
        align=lambda v:(v+63)//64*64;pad=(n['channels']+7)//8*8
        weight=align(weight)+pad*n['terms'];bias=align(bias)+pad*4;mult=align(mult)+pad*4;relu+=pad//8
    assert weight<=12608 and bias<=15040 and mult<=17472 and relu<=17600,(weight,bias,mult,relu)
    old=CoarseSkipNet().eval();net=Surface32Net().eval();net.expand_from(old);x=torch.rand(1,3,64,96)*2-1
    with torch.no_grad():err=float((old(x)-net(x)).abs().max())
    assert err<2e-6
    write_json(out/'audit.json',dict(status='FINITE_ALLOCATION_ONLY',width=a.width,macs=plan()['macs'],nodes=len(plan()['nodes']),stripes=rows,weight_end=weight,bias_end=bias,multiplier_end=mult,relu_end=relu,expansion_max_error=err,source_sha256=file_hash(__file__),fpga=False,style=False))


if __name__=='__main__':main()
