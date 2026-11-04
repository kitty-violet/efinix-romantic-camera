"""Counterfactual weight-only test of premature hard decisions at180p.

Change existing PW slopes around63.5, without touching the final360/720
decoder, graph or weight capacity. These are actual B1 coefficient changes,
not a runtime filter; every candidate remains explicitly unaccepted.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_monotone_pair_decoder import make_board
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_white_rims_grid import measures
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    variants=dict(base={},pw1_075={'res3_conv1__pw':.75},pw1_050={'res3_conv1__pw':.5},pw2_067={'res3_conv2__pw':2/3},both={'res3_conv1__pw':.75,'res3_conv2__pw':2/3})
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2));expected=fragment(base,z,'up1_low_dw')['rgb_head']
    rows=m.load_data(a.data)['validation'];records=[];boards=[{} for _ in range(len(variants)+1)]
    for row in rows:boards[0][row['id']]=m.t.r.c.rgb_image(torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy()).float()/128-1)
    for j,(name,changes) in enumerate(variants.items(),1):
        net=copy.deepcopy(base)
        for key,factor in changes.items():
            net.weight[key].mul_(factor);net.bias[key].copy_(63.5+(net.bias[key]-63.5)*factor)
        assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'],expected)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),manual_early_threshold_slopes=changes,parent_checkpoint=a.checkpoint,parent_sha256=file_hash(a.checkpoint),step=0,added_training_steps=0,visual_acceptance=False)
        torch.save(saved,dest/(name+'.pt'));rr=[]
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            nodes=net(x/128,all_nodes=True);replay,_=actual(x,True);assert all(torch.equal(v,replay[k]) for k,v in nodes.items())
            p=nodes['rgb_head']/128;profile,pv=reference_boundary_profile(p,t,labels);white,_=measures(p*128+128,t*128+128,x+128)
            rr.append(dict(id=row['id'],mae=float((p-t).abs().mean()*128),profile=float(profile),**{k:float(v) for k,v in pv.items()},**white))
            im=m.t.r.c.rgb_image(p[0]);im.save(dest/(row['id']+'_'+name+'.png'));boards[j][row['id']]=im
        records.append(dict(variant=name,changes=changes,records=rr,mean_mae=sum(r['mae'] for r in rr)/len(rr),mean_profile=sum(r['profile'] for r in rr)/len(rr)))
        print('EARLY_THRESHOLD_PROBE',name,records[-1]['mean_mae'],records[-1]['mean_profile'],flush=True)
        write_json(dest/'progress.json',dict(records=records,complete=False))
    make_board(boards,['target']+list(variants),dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,all_variant_six720_all_node_error=0,constant1215_error=0,bytes=20480,
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),added_inference_operations=0,visual_acceptance=False))


if __name__=='__main__':main()
