"""Exact ordered-head reduction: C3 Y/Co/Cg followed by fixed scalar tables.

The trained 32 logits are affine functions of these three components. Their
argmax is the nearest fixed level, so the C32 convolution can be removed.
INT8 output rounding is a NEW approximation and must be tested/trained.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn
from PIL import Image
from .coarse_skip_net import CoarseSkipNet,plan as previous_plan
from .ordinal_coarse_net import OrdinalCoarseNet
from .train_luma_regions import ROOT,RUN,LEVELS,centers,reconstruct,display
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import file_hash,write_json


def plan():
    p=copy.deepcopy(previous_plan());n=p['nodes'][-1];n.update(channels=3,output_shape_hwc=[720,1280,3],activation='clamp_rgb',macs=720*1280*8*3)
    p.update(macs=sum(n['macs'] for n in p['nodes']),output='signed Y-128/Co/Cg; nearest fixed levels then RGB reconstruction');return p


def decode(x):return reconstruct(*[v[(x[:,i,None]-v[None,:,None,None]).abs().argmin(1)] for i,v in enumerate(centers(x))])


def fused_plan():
    result=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=result.pop();assert p['id']==n['inputs'][0]
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        result.append(n)
    result[-1]['id']='rgb_head';return dict(nodes=result)


class ComponentNet(CoarseSkipNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes'];self.layers['rgb']=nn.Sequential(nn.Conv2d(8,3,1))

    @torch.no_grad()
    def from_ordinal(self,source):
        state={k:v for k,v in source.state_dict().items() if '.parametrizations.' not in k}
        h=source.conv('rgb');state['layers.rgb.0.weight']=h.parametrizations.weight.original[:3].detach().clone()
        state['layers.rgb.0.bias']=h.parametrizations.bias.original[:3].detach().clone();self.load_state_dict(state)

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='component-regions-c3-int8-q17-v1',levels=LEVELS,output=plan()['output']);return p


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    torch.set_num_threads(2);out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in s['binding']['code'].items())
    source=OrdinalCoarseNet().eval();source.load_state_dict(s['model']);net=ComponentNet().eval();net.from_ordinal(source)
    data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'));checks=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None];a0=display(source(x));raw=net(x);b=decode(raw);q=decode(torch.round(raw*128).clamp(-128,127)/128)
        different=float((a0!=b).any(1).float().mean());assert different<1e-5,(r['id'],different)
        Image.fromarray(to_rgb(b[0])).save(out/(r['id']+'_fp32.png'));Image.fromarray(to_rgb(q[0])).save(out/(r['id']+'_output_rounding.png'))
        checks.append(dict(id=r['id'],fp32_pixel_change_fraction=different,output_rounding_mae=float((q-b).abs().mean()*128),output_rounding_pixel_change_fraction=float((q!=b).any(1).float().mean())))
        print('COMPONENT_HEAD_CHECK',checks[-1],flush=True)
    binding=copy.deepcopy(s['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    torch.save(dict(model=net.state_dict(),scales=None,mode='fp32_component_export',step=s['step'],config=s['config'],binding=binding,source=str(a.checkpoint),source_sha256=file_hash(a.checkpoint)),out/'fp32_component.pt')
    for f in ['data.json','config.json']:(out/f).write_bytes((a.checkpoint.parent/f).read_bytes())
    write_json(out/'graph.json',plan());write_json(out/'report.json',dict(status='FP32_COMPONENT_HEAD_CHECKED',checks=checks,ideal_macs=plan()['macs'],old_ideal_macs=previous_plan()['macs'],source_sha256=file_hash(a.checkpoint),full_int8=False,rtl=False,fit=False,style_accepted=False))


if __name__=='__main__':main()
