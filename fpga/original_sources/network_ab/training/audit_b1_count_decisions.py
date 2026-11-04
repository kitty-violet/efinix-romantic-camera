"""Audit soft-count stationarity versus actual integer colour decisions on TRAIN crops."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_count_regions import log_count_probability, count_targets
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .build_b1_ordered_palette import GROUPS
from .runtime import write_json,file_hash


def audit(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding']
    pk=palette_key(torch.tensor(state['palette']));xs=[];codes=[];ids=[]
    for suffix,(x,y) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in m.load_data()['training'] if Path(r['video']).stem.endswith(suffix))
        assert row['split']=='train';ids.append(row['id'])
        source=m.t.r.read(row['path'])[y:y+320,x:x+320]
        xs.append(torch.tensor(source.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        codes.append(torch.tensor(region_class_map(row['target'],row['region'],pk)[y:y+320,x:x+320].astype(np.int64)))
    with torch.no_grad():nodes=net(torch.stack(xs),all_nodes=True)
    fields=((nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4).detach()
    targets=[v[...,20:-20,20:-20] for v in count_targets(torch.stack(codes))]
    records=[]
    for name,(start,end),truth in zip(('Y','Co','Cg'),GROUPS,targets):
        z=fields[:,start:end].clone().requires_grad_(True)
        pure=truth.max(1).values>.999;wanted=truth.argmax(1)
        actual=(z>=0).sum(1);expected=z.sigmoid().sum(1)
        wrong=(actual!=wanted)&pure
        per_pixel=-(truth*log_count_probability(z)).sum(1)
        grad=torch.autograd.grad(per_pixel.sum(),z)[0]
        # Correcting a too-low hard count requires some currently-off field
        # to move up. Check near-stationary wrong decisions, not mean bit accuracy.
        near_stationary=grad.abs().amax(1)<.02
        expected_right=(expected-wanted).abs()<.25
        n=wrong.sum().clamp_min(1)
        records.append(dict(group=name,pure_pixels=int(pure.sum()),hard_wrong_fraction=float(wrong.sum()/pure.sum()),
            wrong_but_expected_within_quarter=float((wrong&expected_right).sum()/n),
            wrong_near_stationary_fraction=float((wrong&near_stationary).sum()/n),
            wrong_mean_gradient_max=float(grad.abs().amax(1)[wrong].mean()),
            wrong_mean_loss=float(per_pixel[wrong].mean().detach()),
            correct_mean_loss=float(per_pixel[pure&~wrong].mean().detach())))
    return dict(checkpoint=str(path),sha256=file_hash(path),training_ids=ids,records=records,
        diagnostic_only=True,threshold_near_stationary=.02,not_a_proof_of_primary_image_cause=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);r=audit(a.checkpoint);write_json(a.out,r);print(r,flush=True)
