"""Sorting intervention isolates invalid-order sensitivity of radix decoding.

Sorting is not deployable on current B1. It preserves every pixel's coordinate
sum and does not use targets to alter predictions. Only a diagnostic.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import GROUPS30
from .probe_b1_monotone_pair_decoder import make_board
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);raw=net.export();dest=Path(a.out);dest.mkdir(exist_ok=False);boards=[{},{},{}];records=[]
    for row in m.load_data(a.data)['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        nodes=net(x,all_nodes=True);z=nodes['up1_low_dw'];ordered=z.clone();viol=[]
        for lo,hi in GROUPS30:
            ordered[:,lo:hi]=z[:,lo:hi].sort(1,descending=True).values
            assert torch.equal(ordered[:,lo:hi].sum(1),z[:,lo:hi].sum(1))
            viol.append(torch.diff(z[:,lo:hi],dim=1).amax(1))
        bad=torch.stack(viol).amax(0)>1
        pred=nodes['rgb_head']/128;corrected=fragment(net,ordered,'up1_low_dw')['rgb_head']/128
        for board,name,p in zip(boards,('target','actual','order_diagnostic'),(t,pred,corrected)):
            profile,_=reference_boundary_profile(p,t,labels)
            rec=dict(id=row['id'],mode=name,mae=float((p-t).abs().mean()*128),profile=float(profile),rank_violation_pixel_fraction=float(bad.float().mean()),rgb_changed_fraction=float(((corrected-pred).abs().amax(1)>1/128).float().mean()))
            records.append(rec);board[row['id']]=m.t.r.c.rgb_image(p[0]);board[row['id']].save(dest/f'{row["id"]}_{name}.png');print('ORDINAL_CODE_ORDER',rec,flush=True)
    assert net.export()==raw
    make_board(boards,['target','actual prediction','sum-preserving sort DIAGNOSTIC'],dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,sorting_not_in_B1=True,weights_unchanged=True,prediction_sort_uses_target=False,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))


if __name__=='__main__':main()
