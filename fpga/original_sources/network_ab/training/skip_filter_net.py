"""Learned native-skip filtering, entirely in existing DW/PW operations."""
import argparse,copy,sys
from collections import Counter
import torch
from torch import nn
from .context_region_net import ContextNet,plan as context_plan
from .region_detail_target import ROOT,RUN
from .probe_boundary_budget import HW,tile
from .runtime import write_json,file_hash


def plan():
    old=context_plan();nodes=[]
    for n in old['nodes']:
        n=copy.deepcopy(n)
        if n['id']=='mix.pw':
            for name,kind,parent,k in [('native.dw','dw3x3','stem',3),('native.pw','pw1x1','native.dw',1)]:
                terms=k*k if k==3 else 8
                nodes.append(dict(id=name,kind=kind,inputs=[parent],input_channels=8,channels=8,stride=1,
                    input_shape_hwc=[720,1280,8],output_shape_hwc=[720,1280,8],kernel=k,groups=8 if k==3 else 1,
                    activation='relu',terms=terms,macs=720*1280*8*terms))
        if n['id']=='mix_add':n['inputs'][1]='native.pw'
        nodes.append(n)
    return dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False)


def fused_plan():
    nodes=[]
    for n in plan()['nodes']:
        n=copy.deepcopy(n)
        if n['kind']=='residual_add':
            p=nodes.pop();assert p['id']==n['inputs'][0] and p['kind']=='pw1x1'
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        nodes.append(n)
    nodes[-1]['id']='rgb_head';return dict(nodes=nodes)


class SkipFilterNet(ContextNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes']
        for n in self.nodes:
            if not n['id'].startswith('native.'):continue
            c=nn.Conv2d(8,8,n['kernel'],groups=n['groups']);parts=[nn.ReplicationPad2d(1)] if n['kernel']==3 else []
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts,c,nn.ReLU())
            with torch.no_grad():
                c.bias.zero_();c.weight.zero_()
                if n['kernel']==3:c.weight[:,:,1,1]=1
                else:c.weight[:,:,0,0].copy_(torch.eye(8))

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='learned-native-context-luma-int8-q17-v1');return p


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=48);a=p.parse_args()
    out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    graph=fused_plan();counts=Counter();rows=[]
    for i,x in enumerate(range(0,1280,a.width)):
        try:
            r=compile_tile(graph,dict(name='research_native_context'),tile(graph,i,x,min(a.width,1280-x)),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        except Exception as e:
            write_json(out/'audit.json',dict(status='ALLOCATION_FAIL',x=x,width=a.width,error=str(e)));raise
        counts.update(q['kind'] for q in r['commands']);rows.append(dict(x=x,high=r['finite_high_water'],peak=r['peak_live_padded_bytes']))
        print('NATIVE_BUDGET',i+1,flush=True)
    write_json(out/'audit.json',dict(status='FINITE_ALLOCATION_ONLY',width=a.width,ideal_macs=plan()['macs'],counts=dict(counts),stripes=rows,source_sha256=file_hash(__file__),style=False,fit=False,timing=False))

if __name__=='__main__':main()
