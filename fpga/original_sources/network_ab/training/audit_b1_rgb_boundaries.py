"""Actual-byte review of RGB boundary alignment; metrics are not acceptance.

Report both the old independent-channel match and a shared-vector match, plus
colour departure from the local target edge segment. The last measure also
includes colour bias and junction complexity; it is not a pure halo detector.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss
from .train_b1_region_formation import formation_loss
from .runtime import write_json,file_hash


def off_segment(p,t):
    values=[]
    for dim in (-1,-2):
        step=torch.diff(t,dim=dim);mask=step.abs().amax(1)>6/128
        pad=(4,4,0,0) if dim==-1 else (0,0,4,4)
        tt=F.pad(t,pad,mode='replicate');n=step.shape[dim]
        a=tt.narrow(dim,0,n);b=tt.narrow(dim,9,n);direction=b-a
        norm=direction.square().sum(1,keepdim=True).clamp_min(1e-5)
        pp=F.pad(p,pad,mode='replicate');samples=[]
        for offset in range(2,8):
            q=pp.narrow(dim,offset,n);alpha=((q-a)*direction).sum(1,keepdim=True)/norm
            distance=(q-a-alpha.clamp(0,1)*direction).abs().mean(1)
            samples.append(distance[mask].mean())
        values.append(torch.stack(samples).mean())
    return torch.stack(values).mean()*128


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);rows=m.load_data(args.data)['validation'];results=[]
    for checkpoint in args.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');records=[]
        for row in rows:
            def tensor(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
            x,t=tensor(row['path']),tensor(row['target']);p=actual(x*128)/128
            lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            _,old=formation_loss(p,t,lab)
            records.append(dict(id=row['id'],joint_vector=float(rgb_edge_vector_loss(p,t)),
                old_independent=float(old['missing_sharp_boundary']),off_segment_rgb=float(off_segment(p,t)),target_off_segment_rgb=float(off_segment(t,t))))
        mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'}
        results.append(dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),records=records,mean=mean));print('RGB_BOUNDARY_REVIEW',str(checkpoint),mean,flush=True)
    write_json(args.out,dict(results=results,actual_bytes=True,scope='Six previously used validation images. No blind-test or visual-acceptance claim.'))


if __name__=='__main__':main()
