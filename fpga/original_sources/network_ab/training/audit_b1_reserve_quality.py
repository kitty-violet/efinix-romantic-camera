"""Actual integer replay on reserved scenes, not previously blind test claims."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .probe_b1_encoder_gauge import load
from .audit_b1_white_rims_grid import measures
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def scalar_means(records):
    return {k:sum(r[k] for r in records)/len(records)
            for k,v in records[0].items() if isinstance(v,(int,float))}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    data=m.load_data(a.data);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');records=[]
    for row in data['reserved_test']:
        for path,sha in [('path','sha256'),('target','target_sha256'),('region','region_sha256')]:assert file_hash(row[path])==row[sha]
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target']);labels=m.load_region(row['region']).astype(np.int64)
        x=torch.from_numpy(source.transpose(2,0,1).copy())[None].float()-128;t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x/128,all_nodes=True);replay,_=actual(x,True);assert all(torch.equal(v,replay[k]) for k,v in nodes.items())
        p=nodes['rgb_head']/128;lab=torch.from_numpy(labels)[None]
        _,v=joint_region_loss(p,t,lab);_,profile=reference_boundary_profile(p,t,lab);bright,_=measures(p*128+128,t*128+128,x+128)
        delta=(p-t)*128;r,g,b=delta.unbind(1)
        record=dict(id=row['id'],mae=float(delta.abs().mean()),bias_Co=float(((r-b)/2).mean()),bias_Cg=float(((-r+2*g-b)/4).mean()),
            **{k:float(z) for k,z in v.items()},**{k:float(z) for k,z in profile.items()},**bright)
        records.append(record);m.t.r.c.rgb_image(p[0]).save(dest/(row['id']+'_integer720.png'))
        write_json(dest/'progress.json',dict(records=records,complete=False,checkpoint_sha256=file_hash(a.checkpoint)))
        print('RESERVE_ACTUAL_QUALITY',record['id'],record['mae'],record['rgb_profile_shape'],record['rgb_profile_contrast'],flush=True)
    assert len(records)==10
    write_json(dest/'review.json',dict(records=records,mean=scalar_means(records),
        all_ten720_all_node_error=0,bytes=len(raw),checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),
        scope='reserved scenes previously used in reconstruction work; not an untouched blind test',used_for_training_here=False,visual_acceptance=False))


if __name__=='__main__':
    assert scalar_means([dict(id='a',mae=2.,grid={'x':[1]}),dict(id='b',mae=4.,grid={'x':[2]})])==dict(mae=3.)
    main()
