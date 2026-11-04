"""Wide regional context with the detail connection at 360p, not 720p.

Removes the long-lived high-resolution skip proven to copy surface grain.
Only existing operator types; keeps native 720p learned output and palette.
"""
import argparse,copy,sys
from collections import Counter
import torch
from torch import nn
from .wide_context_net import WideContextNet,plan as wide_plan
from .probe_boundary_budget import HW,tile
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json


def plan():
    nodes=[];template={n['id']:n for n in wide_plan()['nodes']}
    for old in wide_plan()['nodes']:
        n=copy.deepcopy(old)
        if n['id'] in ('native.dw','native.pw','mix_add'):continue
        if n['id']=='context.up.pw':
            # The skip is ready before its fusing PW, and is retained at 360p.
            s=copy.deepcopy(template['native.pw']);s['inputs']=['down.dw']
            for key in ['input_shape_hwc','output_shape_hwc']:s[key]=[360,640,8]
            s['macs']//=4;nodes.append(s)
        nodes.append(n)
        if n['id']=='context.up.pw':
            n['activation']='none';s=copy.deepcopy(template['mix_add']);s.update(id='context.skip_add',inputs=['context.up.pw','native.pw'])
            for key in ['input_shape_hwc','output_shape_hwc']:s[key]=[360,640,8]
            nodes.append(s)
        if n['id']=='up_nn':n['inputs']=['context.skip_add']
        if n['id']=='finish.dw':n['inputs']=['mix.pw']
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


class CoarseSkipNet(WideContextNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes'];del self.layers['native__dw']
        self.layers['context__up__pw']=nn.Sequential(self.conv('context.up.pw'))

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='coarse-skip-wide-luma-int8-q17-v1');return p


def audit(out,width):
    sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    out.mkdir(exist_ok=False);graph=fused_plan();counts=Counter();rows=[];byid={n['id']:n for n in graph['nodes']};macs=0
    for i,x in enumerate(range(0,1280,width)):
        try:r=compile_tile(graph,dict(name='research_coarse_skip'),tile(graph,i,x,min(width,1280-x)),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        except Exception as e:write_json(out/'audit.json',dict(status='ALLOCATION_FAIL',x=x,width=width,error=str(e)));raise
        counts.update(q['kind'] for q in r['commands']);rows.append(dict(x=x,high=r['finite_high_water'],peak=r['peak_live_padded_bytes']))
        for q in r['commands']:
            if q['kind']=='compute':n=byid[q['node']];macs+=q['pixels']*q['rows']*n['channels']*n['terms']
        print('COARSE_SKIP_BUDGET',i+1,flush=True)
    write_json(out/'audit.json',dict(status='FINITE_ALLOCATION_ONLY',width=width,ideal_macs=plan()['macs'],scheduled_macs=macs,counts=dict(counts),stripes=rows,source_sha256=file_hash(__file__),style=False,fit=False,timing=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=48);a=p.parse_args();audit(ROOT/'runs'/RUN/a.name,a.width)
