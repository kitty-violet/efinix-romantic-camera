"""TRAIN-only field-gradient scale check for the new coincidence term."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .probe_b1_encoder_gauge import load
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_coincident_distance import coincident_loss
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);data=m.load_data(a.data)
    setup=state['binary_coverage720'];levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    records=[]
    for video in sorted({r['video'] for r in data['training']}):
        options=[r for r in data['training'] if r['video']==video];row=options[len(options)//2]
        sl=np.s_[200:520,480:800];source=m.t.r.read(row['path'])[sl]
        x=torch.from_numpy(source.transpose(2,0,1).copy())[None].float()/128-1
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        truth=distance_fields(code,8)
        with torch.no_grad():field=net(x,all_nodes=True)['up1_low_dw']
        z=field[...,10:-10,10:-10].detach().requires_grad_();t=truth[...,10:-10,10:-10]
        full,band=distance_objective(z,t);base=16*full+8*band+10*rank_loss(z);shared=8*coincident_loss(z,t)
        g=torch.autograd.grad(base,z,retain_graph=True)[0];h=torch.autograd.grad(shared,z)[0]
        records.append(dict(id=row['id'],base_loss=float(base.detach()),shared_loss=float(shared.detach()),
          base_norm=float(g.norm()),shared_norm=float(h.norm()),ratio=float(h.norm()/g.norm().clamp_min(1e-12)),
          cosine=float((g*h).sum()/(g.norm()*h.norm()).clamp_min(1e-12))))
        print('COINCIDENT_GRADIENT_SCALE',records[-1],flush=True)
    write_json(Path(a.out),dict(records=records,scope='7 TRAIN source crops; field gradients only, excluding RGB and temporal terms',
      shared_weight=8,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))


if __name__=='__main__':main()
