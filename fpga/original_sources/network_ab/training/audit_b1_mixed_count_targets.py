"""Measure a concrete likelihood mismatch, without assuming it is the root cause.

A 2x2 footprint can contain only colours0 and7 in equal area. Independent
Bernoulli fields cannot give probability .5 to count0 and .5 to count7.
This diagnostic quantifies how often non-adjacent count mixtures occur in
TRAIN. It changes neither labels nor inference.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_count_regions import count_targets, log_count_probability
from .fit_b1_sixteen_regions import palette_key, region_class_map
from .runtime import write_json, file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    pk=palette_key(torch.tensor(state['palette']));records=[]
    for row in m.load_data()['training']:
        codes=torch.tensor(region_class_map(row['target'],row['region'],pk).astype(np.int64))[None]
        targets=count_targets(codes);mixed=torch.stack([v.amax(1)<.999 for v in targets]).any(0)
        gaps=[]
        for v in targets:
            present=v>0
            span=present.sum(1)
            index=torch.arange(v.shape[1])[None,:,None,None]
            lo=torch.where(present,index,v.shape[1]).amin(1)
            hi=torch.where(present,index,-1).amax(1)
            gaps.append((hi-lo+1)>span)
        records.append(dict(id=row['id'],mixed_fraction=float(mixed.float().mean()),
                            noncontiguous_level_support_fraction=float(torch.stack(gaps).any(0).float().mean())))
    # Symmetric endpoint cross entropy is minimized independently by p(bit)=.5.
    z=torch.zeros(1,7,1,1,dtype=torch.float64,requires_grad=True)
    p=log_count_probability(z).exp().flatten()
    loss=-(p[0].log()+p[-1].log())/2
    loss.backward();assert torch.count_nonzero(z.grad)==0
    assert abs(float(p[1:-1].sum().detach())-.984375)<1e-12
    result=dict(checkpoint=args.checkpoint,checkpoint_sha256=file_hash(args.checkpoint),
                training_frames=len(records),records=records,
                mean_mixed_fraction=float(np.mean([r['mixed_fraction'] for r in records])),
                mean_noncontiguous_support_fraction=float(np.mean([r['noncontiguous_level_support_fraction'] for r in records])),
                synthetic_target=[.5,0,0,0,0,0,0,.5],optimal_independent_distribution=p.detach().tolist(),
                synthetic_intermediate_mass=float(p[1:-1].sum().detach()),
                diagnostic_only=True,proof_of_real_image_root_cause=False)
    write_json(args.out,result);print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':main()
