"""Diagnose whether ordered predicates split one target contour into bands.

The local target-dependent pooling below is NOT deployable inference. It
preserves each selected group's predicted mean while aligning its predicates,
to isolate relative boundary disagreement from global colour error.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .probe_b1_monotone_pair_decoder import make_board
from .build_b1_thirty_binary_regions import GROUPS30
from .train_b1_ternary_regions import region_codes
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def coincidence(truth):
    band=(F.max_pool2d(truth,5,1,2)+F.max_pool2d(-truth,5,1,2))>2
    return [((truth[:,lo:hi-1]-truth[:,lo+1:hi]).abs()<.25)&band[:,lo:hi-1]&band[:,lo+1:hi] for lo,hi in GROUPS30]


def diagnostic_pool(field,truth):
    result=field.clone();gaps=[];masks=coincidence(truth)
    for (lo,hi),same in zip(GROUPS30,masks):
        ids=F.pad((~same).long().cumsum(1),(0,0,0,0,1,0))
        p=field[:,lo:hi];sums=torch.zeros_like(p).scatter_add(1,ids,p)
        count=torch.zeros_like(p).scatter_add(1,ids,torch.ones_like(p)).clamp_min(1)
        pooled=(sums/count).gather(1,ids)
        assert (p.sum(1)-pooled.sum(1)).abs().max()<.002
        result[:,lo:hi]=pooled
        gaps.append(((p[:,:-1]-p[:,1:]).abs()*same).sum()/same.sum().clamp_min(1))
    return result,torch.stack(gaps).mean(),float(torch.cat(masks,1).float().mean())


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    rows=m.load_data(a.data)['validation'];boards=[{},{},{}];records=[]
    for row in rows:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        truth=distance_fields(code,8);nodes=net(x,all_nodes=True)
        pooled,gap,fraction=diagnostic_pool(nodes['up1_low_dw'],truth)
        actual=nodes['rgb_head']/128;corrected=fragment(net,net.round(pooled),'up1_low_dw')['rgb_head']/128
        before,terms=reference_boundary_profile(actual,target,labels);after,newterms=reference_boundary_profile(corrected,target,labels)
        records.append(dict(id=row['id'],ordered_coincident_gap=float(gap),active_pair_fraction=fraction,
          before_profile=float(before),after_profile=float(after),before_shape=float(terms['rgb_profile_shape']),after_shape=float(newterms['rgb_profile_shape']),
          before_mae=float((actual-target).abs().mean()*128),after_mae=float((corrected-target).abs().mean()*128)))
        for b,p in zip(boards,(target,actual,corrected)):b[row['id']]=m.t.r.c.rgb_image(p[0])
        print('COINCIDENT_EDGE_DIAGNOSTIC',records[-1],flush=True)
    make_board(boards,['target','actual source prediction','GT contour grouping diagnostic ONLY'],dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),
      extra_inference_proposed=False,diagnostic_uses_ground_truth=True,not_deployable=True,visual_acceptance=False))


if __name__=='__main__':main()
