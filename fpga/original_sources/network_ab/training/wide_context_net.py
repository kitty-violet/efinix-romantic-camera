"""Third-scale regional context using the same DW/PW/nearest SRAM engine."""
import argparse,copy,sys
from collections import Counter
import torch
from torch import nn
from .skip_filter_net import SkipFilterNet,plan as previous_plan
from .probe_boundary_budget import HW,tile
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json


def plan():
    nodes=[]
    def add(name,kind,parent,h,channels,stride=1):
        k=3 if kind=='dw3x3' else 1 if kind=='pw1x1' else 0
        ho=h*2 if kind=='nearest_x2' else h//stride;groups=channels if k==3 else 1;terms=channels*k*k//groups if k else 0
        nodes.append(dict(id=name,kind=kind,inputs=[parent],input_channels=channels,channels=channels,stride=stride,
            input_shape_hwc=[h,h*16//9,channels],output_shape_hwc=[ho,ho*16//9,channels],kernel=k,groups=groups,
            activation='none' if not k else 'relu',terms=terms,macs=ho*(ho*16//9)*channels*terms))
    for old in previous_plan()['nodes']:
        n=copy.deepcopy(old)
        if n['id']=='res1_conv1.dw':
            add('wide.down.dw','dw3x3','context.down.pw',180,16,2)
            add('wide.down.pw','pw1x1','wide.down.dw',90,16)
        if n['id'].startswith(('res1_','decode.')):
            n['inputs']=['wide.down.pw' if p=='context.down.pw' else p for p in n['inputs']]
            for key in ['input_shape_hwc','output_shape_hwc']:n[key]=[n[key][0]//2,n[key][1]//2,n[key][2]]
            n['macs']//=4
        if n['id']=='context.up':
            add('wide.up','nearest_x2','decode.pw',90,8)
            add('wide.up.dw','dw3x3','wide.up',180,8)
            add('wide.up.pw','pw1x1','wide.up.dw',180,8)
            n['inputs']=['wide.up.pw']
        nodes.append(n)
    assert len(nodes)<=29
    return dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False)


def fused_plan():
    nodes=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=nodes.pop();assert p['id']==n['inputs'][0] and p['kind']=='pw1x1'
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        nodes.append(n)
    nodes[-1]['id']='rgb_head';return dict(nodes=nodes)


class WideContextNet(SkipFilterNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes']
        for n in self.nodes:
            if not n['id'].startswith('wide.') or not n['kernel']:continue
            c=nn.Conv2d(n['input_channels'],n['channels'],n['kernel'],n['stride'],groups=n['groups'])
            parts=([nn.ReplicationPad2d(1)] if n['kernel']==3 else [])+[c,nn.ReLU()]
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)
            with torch.no_grad():
                c.bias.zero_()
                if n['kernel']==3:c.weight.fill_(1/9)
                else:c.weight.zero_();c.weight[:,:,0,0].copy_(torch.eye(c.in_channels))

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='wide-native-context-luma-int8-q17-v1');return p


def audit(out,width):
    sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    out.mkdir(exist_ok=False);graph=fused_plan();counts=Counter();rows=[];byid={n['id']:n for n in graph['nodes']};macs=0
    for i,x in enumerate(range(0,1280,width)):
        try:r=compile_tile(graph,dict(name='research_wide_native'),tile(graph,i,x,min(width,1280-x)),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        except Exception as e:write_json(out/'audit.json',dict(status='ALLOCATION_FAIL',x=x,width=width,error=str(e)));raise
        counts.update(q['kind'] for q in r['commands']);rows.append(dict(x=x,high=r['finite_high_water'],peak=r['peak_live_padded_bytes']))
        for q in r['commands']:
            if q['kind']=='compute':n=byid[q['node']];macs+=q['pixels']*q['rows']*n['channels']*n['terms']
        print('WIDE_BUDGET',i+1,flush=True)
    write_json(out/'audit.json',dict(status='FINITE_ALLOCATION_ONLY',width=width,ideal_macs=plan()['macs'],scheduled_macs=macs,counts=dict(counts),stripes=rows,source_sha256=file_hash(__file__),style=False,fit=False,timing=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=48);a=p.parse_args();audit(ROOT/'runs'/RUN/a.name,a.width)
