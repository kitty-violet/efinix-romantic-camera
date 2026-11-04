"""Measure whether predicted contour-distance fields have shallow transitions.

No rescaling or filter is applied to the actual output. This distinguishes
incorrect distance amplitude from merely shifted zero contours.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def slope(field):
    gx=(field[...,1:-1,2:]-field[...,1:-1,:-2])/2
    gy=(field[...,2:,1:-1]-field[...,:-2,1:-1])/2
    return (gx.square()+gy.square()+1e-8).sqrt()


def summary(norm,mask):
    values=norm[mask];assert len(values)>0
    q=torch.quantile(values,values.new_tensor([.1,.5,.9]))
    return dict(count=len(values),p10=float(q[0]),median=float(q[1]),p90=float(q[2]),below8=float((values<8).float().mean()),above48=float((values>48).float().mean()))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);data=m.load_data(a.data)
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    rows=[]
    for video in sorted({r['video'] for r in data['training']}):
        rr=[r for r in data['training'] if r['video']==video];rows.append(rr[len(rr)//2])
    records=[]
    for row in rows+data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        truth=distance_fields(code,8);field=net(x,all_nodes=True)['up1_low_dw'][:,:30]
        nearby=(F.max_pool2d(truth,5,1,2)+F.max_pool2d(-truth,5,1,2))>2
        pc=field[...,1:-1,1:-1];tc=truth[...,1:-1,1:-1];nearby=nearby[...,1:-1,1:-1]
        pm=(pc-63.5).abs()<24;tm=(tc-63.5).abs()<24
        result=dict(id=row['id'],predicted_zero_band=summary(slope(field),pm&nearby),
            target_zero_band=summary(slope(truth),tm),same_target_band=summary(slope(field),tm),
            predicted_mixed_fraction=float(pm.float().mean()),target_mixed_fraction=float(tm.float().mean()))
        records.append(result);print('DISTANCE_SLOPE_AUDIT',result,flush=True)
    write_json(Path(a.out),dict(records=records,expected_unclipped_plane_slope_per180_cell=32,
        caveat='Finite differences, clipping, corners and small regions alter the ideal slope; these are diagnostics, not hard universal thresholds.',
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),inference_changed=False,visual_acceptance=False))


if __name__=='__main__':main()
