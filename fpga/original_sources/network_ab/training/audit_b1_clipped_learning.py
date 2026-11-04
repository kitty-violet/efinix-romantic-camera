"""CPU derivative audit of clipped activations, using TRAIN crops only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_count_regions import count_targets,count_loss
from .train_b1_ordered_regions import OrderedGradientB1
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.__class__=OrderedGradientB1;net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding']
    pk=palette_key(torch.tensor(state['palette']));xs=[];codes=[];ids=[];data=m.load_data()
    for suffix,(x,y) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix));assert row['split']=='train';ids.append(row['id'])
        source=m.t.r.read(row['path'])[y:y+320,x:x+320]
        xs.append(torch.tensor(source.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        codes.append(torch.tensor(region_class_map(row['target'],row['region'],pk)[y:y+320,x:x+320].astype(np.int64)))
    nodes=net(torch.stack(xs),all_nodes=True)
    truth=[v[...,20:-20,20:-20] for v in count_targets(torch.stack(codes))]
    loss=count_loss((nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4,truth)
    stages=[r for r in net.plan['nodes'] if r['id'].replace('.','__') in net.weight]
    stages=stages[:next(i for i,r in enumerate(stages) if r['id']=='up1_high_dw')+1]
    gradients=torch.autograd.grad(loss,[nodes[r['id']] for r in stages],retain_graph=True)
    records=[]
    with torch.no_grad():
        for r,g in zip(stages,gradients):
            key=r['id'].replace('.','__');scale=(getattr(net,key+'_m')/131072).float().flatten()
            w=net.round(net.weight[key]/scale[:,None,None,None]).clamp(-127,127)
            b=net.round(net.bias[key]/scale)
            x=nodes[r['inputs'][0]]
            if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
            raw=(F.conv2d(x,w,stride=r['stride'],groups=r['groups'])+b[None,:,None,None])*scale[None,:,None,None]
            rounded=net.round(raw);lower=0 if r['activation']=='relu' else -128
            clipped=(rounded<lower)|(rounded>127)
            inward=((rounded<lower)&(g<0))|((rounded>127)&(g>0))
            mass=g.abs().sum().clamp_min(1e-20)
            records.append(dict(node=r['id'],activation_clip_fraction=float(clipped.float().mean()),
                requested_gradient_mass_on_clipped=float((g.abs()*clipped).sum()/mass),
                inward_requested_mass_on_clipped=float((g.abs()*inward).sum()/mass),
                gradient_mass=float(mass)))
    result=dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),train_ids=ids,
        main_count_loss=float(loss.detach()),records=records,
        interpretation='Post-activation derivative requests blocked by existing clamp; not proof that changing backward improves quality',
        weights_optimized=False)
    write_json(a.out,result);print(result,flush=True)


if __name__=='__main__':main()
