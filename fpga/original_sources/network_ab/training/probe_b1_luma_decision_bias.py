"""Bounded geometric calibration of existing14Y decisions, real coefficients.

Shift only the final Y predicate centres by2/4/8 code units (ideal SDF8:
0.25/0.5/1 native pixel). Constant palette colours and kernel slope stay fixed.
Measure mean brightness and rims; constant-colour equality alone is not QA.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_monotone_pair_decoder import make_board
from .build_b1_thirty_binary_regions import code_bits
from .audit_b1_rim_tolerance import rim_terms
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);assert state.get('direct_luminance_planes')
    out=Path(a.out);out.mkdir(exist_ok=False);rows=m.load_data(a.data)['validation'];z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    palette=fragment(base,z,'up1_low_dw')['rgb_head'];boards=[];records=[]
    target_images={r['id']:Image.fromarray(m.t.r.read(r['target'])) for r in rows}
    boards.append(target_images)
    for delta in (0,2,4,8):
        net=copy.deepcopy(base);net.bias['up2_low_dw'][:14]-=11*delta*131071/131072
        assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'],palette)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');images={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            nodes=net(x,all_nodes=True);other,_=actual(net.round(x*128),True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            p=nodes['rgb_head']/128;rim,_=rim_terms(p,t,labels,2);_,v=reference_boundary_profile(p,t,labels)
            luma=lambda q:(q[:,0]+2*q[:,1]+q[:,2])/4
            records.append(dict(delta=delta,id=row['id'],mae=float((p-t).abs().mean()*128),mean_luminance_error=float((luma(p)-luma(t)).mean()*128),
                tight_rim=float(rim),profile_shape=float(v['rgb_profile_shape']),profile_contrast=float(v['rgb_profile_contrast'])))
            images[row['id']]=m.t.r.c.rgb_image(p[0]);images[row['id']].save(out/f'delta{delta}_{row["id"]}.png')
        boards.append(images)
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
        saved.update(model=copy.deepcopy(net.state_dict()),manual_luma_decision_bias_codes=delta,constant_palette_preserved=1215,visual_acceptance=False)
        torch.save(saved,out/f'delta{delta}.pt')
        print('LUMA_DECISION_BIAS',delta,records[-len(rows):],flush=True)
    make_board(boards,['目标','原阈值','亮度阈值+2','亮度阈值+4','亮度阈值+8'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,all_nodes_integer_error=0,constant_palette_error=0,bytes=20480,
        extra_inference_operations=0,visual_acceptance=False,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
