"""Wider learned context using only the existing depthwise/pointwise engine."""
import argparse,copy,json,sys
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from .train_luma_regions import LumaNet
from .train_region_classes import class_plan
from .region_detail_target import ROOT,RUN
from .runtime import write_json,file_hash
from .probe_boundary_budget import HW,tile


def plan():
    old=class_plan()['nodes'];nodes=[]
    def add(name,kind,parents,h,stride=1,activation='relu',channels=16):
        ci=co=channels;k=3 if kind=='dw3x3' else 1 if kind=='pw1x1' else 0
        ho=h*2 if kind=='nearest_x2' else h//stride;groups=ci if kind=='dw3x3' else 1
        nodes.append(dict(id=name,kind=kind,inputs=parents,input_channels=ci,channels=co,stride=stride,
            input_shape_hwc=[h,h*16//9,ci],output_shape_hwc=[ho,ho*16//9,co],kernel=k,groups=groups,
            activation=activation,terms=ci*k*k//groups if k else 0,macs=ho*(ho*16//9)*co*ci*k*k//groups if k else 0))
    for n in old:
        n=copy.deepcopy(n)
        if n['id']=='res1_conv1.dw':
            add('context.down.dw','dw3x3',['down.pw'],360,2)
            add('context.down.pw','pw1x1',['context.down.dw'],180)
        if n['id'].startswith(('res1_','decode.')):
            n['inputs']=['context.down.pw' if p=='down.pw' else p for p in n['inputs']]
            for key in ['input_shape_hwc','output_shape_hwc']:n[key]=[n[key][0]//2,n[key][1]//2,n[key][2]]
            n['macs']//=4
        if n['id']=='up_nn':
            add('context.up','nearest_x2',['decode.pw'],180,activation='none',channels=8)
            add('context.up.dw','dw3x3',['context.up'],360,channels=8)
            add('context.up.pw','pw1x1',['context.up.dw'],360,channels=8)
            n['inputs']=['context.up.pw']
        nodes.append(n)
    return dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False)


def fused_plan():
    result=[]
    for n in plan()['nodes']:
        n=copy.deepcopy(n)
        if n['kind']=='residual_add':
            p=result.pop();assert p['id']==n['inputs'][0] and p['kind']=='pw1x1'
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        result.append(n)
    result[-1]['id']='rgb_head';return dict(nodes=result)


class ContextNet(LumaNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes']
        for n in self.nodes:
            if not n['id'].startswith('context.') or not n['kernel']:continue
            parts=[nn.ReplicationPad2d(1)] if n['kernel']==3 else []
            c=nn.Conv2d(n['input_channels'],n['channels'],n['kernel'],n['stride'],groups=n['groups']);nn.init.kaiming_normal_(c.weight,nonlinearity='relu');nn.init.zeros_(c.bias);parts.append(c)
            if n['activation']=='relu':parts.append(nn.ReLU())
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)
        # Start the new path as positive averaging/downsample and identity/up;
        # all coefficients remain learned. This is not claimed equivalent to
        # the previous sampling graph and must be trained/visually verified.
        with torch.no_grad():
            for name in ['context.down.dw','context.up.dw']:
                c=self.conv(name);c.weight.fill_(1/9);c.bias.zero_()
            for name in ['context.down.pw','context.up.pw']:
                c=self.conv(name);c.weight.zero_();c.bias.zero_();c.weight[:,:,0,0].copy_(torch.eye(c.in_channels))

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='context-luma-int8-q17-v1');return p


def audit(name,width):
    root=ROOT/'runs'/RUN/name;root.mkdir(exist_ok=False);sys.path.insert(0,str(HW))
    from fourway_frame_program import compile_tile
    p=fused_plan();byid={n['id']:n for n in p['nodes']};rows=[];counts=Counter();macs=0
    assert width in (32,64)
    for i,x in enumerate(range(0,1280,width)):
        r=compile_tile(p,dict(name='research_context_luma'),tile(p,i,x,width),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        counts.update(q['kind'] for q in r['commands']);rows.append(dict(x=x,high=r['finite_high_water'],peak=r['peak_live_padded_bytes']))
        for q in r['commands']:
            if q['kind']=='compute':n=byid[q['node']];macs+=q['pixels']*q['rows']*n['channels']*n['terms']
        print('CONTEXT_STRIPE',i+1,flush=True)
    # Same serialized storage packing as coefficient_image, independent of weights.
    cursor=1792
    for n in plan()['nodes']:
        if not n['kernel']:continue
        align=lambda x:(x+63)//64*64
        cursor=align(cursor)+(n['channels']+7)//8*8*n['terms'];assert cursor<=12608
        cursor=align(cursor)+(n['channels']+7)//8*8*4
        cursor=align(cursor)+(n['channels']+7)//8*8*4
        cursor=align(cursor)+(n['channels']+7)//8
    result=dict(status='CONTEXT_FINITE_ALLOCATION_ONLY',ideal_macs=plan()['macs'],scheduled_macs=macs,counts=dict(counts),stripes=rows,coefficient_used_bytes=align(cursor),nodes=len(plan()['nodes']),
        source_sha256=file_hash(__file__),requires_new_descriptors=True,hardware_fit=False,timing=False,style=False)
    write_json(root/'audit.json',result);write_json(root/'graph.json',plan());print(json.dumps({k:v for k,v in result.items() if k!='stripes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=32);a=p.parse_args();audit(a.name,a.width)
