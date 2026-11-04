"""TRAIN-only audit of current objective components and oracle augmentation.

No training, checkpoint modification or proposed inference operation. Report
gradient conflicts rather than inferring them from scalar loss magnitudes.
"""
import argparse, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_coincident_distance import coincident_loss
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_reachable_loss_fixedpoint import objective
from .train_b1_real_region_texture import texture_intervention
from .train_b1_ternary_regions import region_codes
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=OrdinalBandB1
    weights=[]
    for key,w in net.weight.items():
        fixed=key.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) or key in ('down1__dw','down2__dw')
        w.requires_grad_(not fixed);net.bias[key].requires_grad_(not fixed)
        if not fixed:weights.extend((w,net.bias[key]))
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    data=m.load_data(a.data);records=[]
    for video in sorted({r['video'] for r in data['training']}):
        options=[r for r in data['training'] if r['video']==video];row=options[len(options)//2];sl=np.s_[200:520,480:800]
        x=torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        codes=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        truth=distance_fields(codes,8)
        with torch.no_grad():target=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
        cut=lambda z:z[...,40:-40,40:-40]
        nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128;z=nodes['up1_low_dw'][...,10:-10,10:-10];t=truth[...,10:-10,10:-10]
        rgb,values=objective(cut(p),cut(target),cut(labels),reference_boundary_profile)
        full,band=distance_objective(z,t)
        losses=dict(rgb=rgb,field=16*full+8*band,rank=10*rank_loss(z),shared=32*coincident_loss(z,t))
        shifted=net(torch.roll(x,(1,1),(-2,-1)))[0]
        losses['phase']=1.5*F.l1_loss(cut(p),cut(torch.roll(shifted,(-1,-1),(-2,-1))))
        altered=texture_intervention(x,labels,rng=random.Random(719))
        augmented=net(altered)[0]
        losses['texture_invariance']=2*F.smooth_l1_loss(augmented,p,beta=4/128)
        vectors={}
        for name,loss in losses.items():
            grads=torch.autograd.grad(loss,weights,retain_graph=True,allow_unused=True)
            vectors[name]=torch.cat([(torch.zeros_like(w) if g is None else g).flatten() for w,g in zip(weights,grads)])
        base=vectors['field']+vectors['rgb']+vectors['rank']+vectors['shared']
        cosine=lambda g,h:float((g*h).sum()/(g.norm()*h.norm()).clamp_min(1e-12))
        augmentation=[]
        with torch.no_grad():
            for gain in (0.,.4,1.,1.6):
                px=net(texture_intervention(x,labels,gains=gain),all_nodes=True)
                ff,bb=distance_objective(px['up1_low_dw'][...,10:-10,10:-10],t)
                augmentation.append(dict(gain=gain,rgb_mae=float((cut(px['rgb_head']/128)-cut(target)).abs().mean()*128),field=float(ff),band=float(bb)))
        record=dict(id=row['id'],losses={k:float(v.detach()) for k,v in losses.items()},gradient_norms={k:float(v.norm()) for k,v in vectors.items()},
            cosines={f'{k}_vs_supervised':cosine(v,base) for k,v in vectors.items()},
            pairs={f'{k}_vs_{j}':cosine(vectors[k],vectors[j]) for k,j in [('field','rgb'),('shared','rgb'),('shared','field')]},
            augmentation=augmentation)
        records.append(record);print('FULL_OBJECTIVE_AUDIT',record,flush=True)
    write_json(Path(a.out),dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),
        scope='Seven TRAIN 320 crops; raw parameter gradients, not optimizer update or causal training ablation',inference_modified=False,visual_acceptance=False))


if __name__=='__main__':main()
