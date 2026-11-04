"""Target-dependent intervention: align equal/opposite SDF patches across Y/Co/Cg.

This is a causal diagnostic, never inference. Unlike the existing adjacent
within-coordinate test, match all 30 fields and require 3x3 patch agreement.
Keep target residuals so an exact target remains unchanged.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .audit_b1_coincident_ordered_edges import diagnostic_pool
from .probe_b1_monotone_pair_decoder import make_board
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def cross_pool(field,truth):
    t=truth; p=field[:,:30]; delta=p-t
    varying=(F.max_pool2d(t,5,1,2)+F.max_pool2d(-t,5,1,2))>2
    active=varying & ((t-63.5).abs()<55)
    result=[]; count_all=0; active_all=0
    for i in range(30):
        # Agreement at one pixel alone can join unrelated equidistant edges.
        same=(F.max_pool2d((t-t[:,i:i+1]).abs(),3,1,1)<.25)
        opposite=(F.max_pool2d((t+t[:,i:i+1]-127).abs(),3,1,1)<.25)
        same &= active & active[:,i:i+1]; opposite &= active & active[:,i:i+1]
        opposite[:,i]=False; same[:,i]=True
        count=(same|opposite).sum(1,keepdim=True)
        correction=((same.float()-opposite.float())*delta).sum(1,keepdim=True)/count
        result.append(t[:,i:i+1]+correction)
        count_all+=int((count>1).sum()); active_all+=count.numel()
    return torch.cat(result+[field[:,30:]],1).clamp(0,127),count_all/max(1,active_all)


def regression():
    ramp=(63.5+torch.arange(-16,16)*20).clamp(0,127)[None,None,None].expand(1,30,24,32).clone()
    ramp[:,14:22]=127-ramp[:,14:22]
    assert torch.equal(cross_pool(ramp,ramp)[0],ramp)
    shifted=torch.roll(ramp,1,-1)
    assert (cross_pool(shifted,ramp)[0]-shifted).abs().max()<1e-5
    wrong=ramp.clone();wrong[:,14:22]=torch.roll(wrong[:,14:22],2,-1)
    before=(wrong-ramp).abs().mean(); after=(cross_pool(wrong,ramp)[0]-ramp).abs().mean()
    assert after<=before+1e-5
    print('CROSS_COLOUR_POOL_PASS target_fixed/common_shift_fixed/opposite_contour',flush=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);regression();state,net=load(a.checkpoint);raw=net.export();dest=Path(a.out);dest.mkdir(exist_ok=False)
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    boards=[{},{},{},{}];records=[]
    for row in m.load_data(a.data)['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        codes=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        truth=distance_fields(codes,8);nodes=net(x,all_nodes=True);p=nodes['up1_low_dw']
        within,_,_=diagnostic_pool(p,truth);cross,fraction=cross_pool(p,truth)
        outputs=[target,nodes['rgb_head']/128]+[fragment(net,net.round(z),'up1_low_dw')['rgb_head']/128 for z in (within,cross)]
        for board,name,out in zip(boards,('target','actual','GT_within','GT_cross'),outputs):
            profile,terms=reference_boundary_profile(out,target,labels)
            rec=dict(id=row['id'],mode=name,mae=float((out-target).abs().mean()*128),profile=float(profile),active_fraction=fraction)
            records.append(rec);im=m.t.r.c.rgb_image(out[0]);board[row['id']]=im;im.save(dest/f'{row["id"]}_{name}.png')
            print('CROSS_COLOUR_CONTOUR',rec,flush=True)
    assert net.export()==raw
    make_board(boards,['target','actual source prediction','GT within-coordinate ONLY','GT cross-coordinate ONLY'],dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),target_dependent_diagnostic=True,weights_unchanged=True,not_deployable=True,visual_acceptance=False))


if __name__=='__main__':main()
