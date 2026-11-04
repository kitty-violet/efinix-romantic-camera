"""TRAIN-only per-channel operating-range and diversity measurements.

These finite-sample quantities do not prove graph capacity or gradient failure.
They distinguish almost-linear positive relays from actual nonlinear gates.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);_,net=load(a.checkpoint);data=m.load_data(a.data)
    picks=[]
    for video in sorted({r['video'] for r in data['training']}):
        rr=[r for r in data['training'] if r['video']==video]
        for pos in sorted(set((0,len(rr)//2,len(rr)-1))):picks.append(rr[pos])
    values={};records=[]
    for row in picks:
        x=torch.from_numpy(m.t.r.read(row['path'])[200:520,480:800].transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True)
        for layer in net.plan['nodes']:
            key=layer['id'];z=nodes[key]
            margin=max(2,40*z.shape[-1]//320);z=z[0,:,margin:-margin,margin:-margin].flatten(1)
            ids=torch.linspace(0,z.shape[-1]-1,512).long();values.setdefault(key,[]).append(z[:,ids])
    for layer in net.plan['nodes']:
        key=layer['id'];z=torch.cat(values[key],1).double();mean=z.mean(1);d=z-mean[:,None]
        eig=torch.linalg.eigvalsh(d@d.T/d.shape[1]).clamp_min(0);rat=eig/eig.sum().clamp_min(1e-20)
        records.append(dict(node=key,activation=layer['activation'],channels=len(z),
            effective_covariance_rank=float((-(rat*rat.clamp_min(1e-20).log()).sum()).exp()),
            positive_unclipped_fraction=((z>0)&(z<127)).double().mean(1).tolist(),
            zero_fraction=(z==0).double().mean(1).tolist(),high_clip_fraction=(z>=127).double().mean(1).tolist(),
            low_clip_fraction=(z<=-128).double().mean(1).tolist(),mean=mean.tolist(),std=z.std(1).tolist()))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    write_json(out,dict(checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),calibration_ids=[r['id'] for r in picks],
        records=records,interpretation='Finite TRAIN operating ranges only. Correlation/rank/dead-channel counts are not an information-theory capacity bound.'))
    print('FEATURE_OPERATING_RANGES',[(r['node'],round(r['effective_covariance_rank'],2),
        sum(v>.99 for v in r['positive_unclipped_fraction']),sum(v>.99 for v in r['zero_fraction'])) for r in records],flush=True)


if __name__=='__main__':main()
