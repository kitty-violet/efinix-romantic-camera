"""Check whether fixed per-row Q17 ranges oppose the RGB training objective.

This is a TRAIN-crop diagnostic of the RGB/region objective (not CE, phase,
temporal or Adam momentum). Near-bound coefficients alone do not prove a root
cause. No optimizer or parameter update is performed.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_ternary_regions import TernaryB1
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss
from .train_b1_region_tail_flatness import region_tail_flatness
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.__class__=TernaryB1
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
    fixed=set(('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'))|set(state.get('encoder_antialias_keys',[]))
    for key in fixed:net.weight[key].requires_grad_(False);net.bias[key].requires_grad_(False)
    rows=[];seen=set()
    for row in m.load_data(args.data)['training']:
        if row['video'] not in seen:seen.add(row['video']);rows.append(row)
    records=[]
    for row in rows:
        def tensor(path):return torch.from_numpy(m.t.r.read(path)[200:520,480:800].transpose(2,0,1).copy())[None].float()/128-1
        x,t=tensor(row['path']),tensor(row['target']);labels=torch.from_numpy(m.load_region(row['region'])[240:480,520:760].astype(np.int64))[None]
        net.zero_grad();p=net(x)[0][...,40:-40,40:-40];t=t[...,40:-40,40:-40]
        region,v=joint_region_loss(p,t,labels)
        loss=.7*region+4*v['region_flatness']+v['adjacent_region_contrast']+2*rgb_edge_vector_loss(p,t)+4*region_tail_flatness(p,labels)
        loss.backward();per=[]
        for key,w in net.weight.items():
            if not w.requires_grad:continue
            scale=(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
            boundary=w.detach().abs()/scale>=126.5
            outward=(w.detach()*w.grad)<0
            per.append(dict(node=key,near_bound=int(boundary.sum()),outward_bound=int((boundary&outward).sum()),
                outward_gradient_mass=float((w.grad.abs()*boundary*outward).sum()),all_gradient_mass=float(w.grad.abs().sum())))
        records.append(dict(id=row['id'],loss=float(loss.detach()),layers=per))
    summary={k:float(np.mean([sum(v[k] for v in r['layers']) for r in records])) for k in ('near_bound','outward_bound','outward_gradient_mass','all_gradient_mass')}
    write_json(args.out,dict(checkpoint_sha256=file_hash(args.checkpoint),training_only=True,parameter_updates=0,
        objective='RGB/region/tail only; excludes count CE, temporal and optimizer momentum',records=records,mean=summary))
    print('ENCODER_RANGE_DIAGNOSTIC',summary,flush=True)


if __name__=='__main__':main()
