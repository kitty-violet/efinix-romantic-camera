"""Compare saved phase heads with the zero-residual baseline on real sources."""
import argparse
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint)
    heads=nn.ModuleDict(dict(phase2=nn.Conv2d(16,4,1),phase4=nn.Conv2d(32,16,1)));heads.load_state_dict(state['heads'])
    data=m.load_data(a.data);selected=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];selected.append(('train',rows[len(rows)//2]))
    selected.extend(('seen_validation',r) for r in data['validation']);records=[]
    for split,row in selected:
        source=m.t.r.read(row['path'])[200:520,480:800]
        x=torch.from_numpy(source.transpose(2,0,1).copy())[None].float()/128-1;nodes=net(x,all_nodes=True)
        y=(x[:,:1]+2*x[:,1:2]+x[:,2:3])/4;res={}
        for scale,node,key in ((2,'down1.pw','phase2'),(4,'down2.pw','phase4')):
            target=F.pixel_unshuffle(y,scale);target-=target.mean(1,keepdim=True)
            q=heads[key](nodes[node]/64-1);q-=q.mean(1,keepdim=True);cut=40//scale
            target=target[...,cut:-cut,cut:-cut];q=q[...,cut:-cut,cut:-cut]
            zero=float(target.square().mean());error=float((q-target).square().mean())
            res[key]=dict(zero_mse=zero,head_mse=error,recovered_energy_fraction=1-error/max(zero,1e-12))
        records.append(dict(id=row['id'],split=split,values=res))
    totals={}
    for split in ('train','seen_validation'):
        totals[split]={}
        for key in ('phase2','phase4'):
            v=[r['values'][key] for r in records if r['split']==split]
            totals[split][key]=1-sum(r['head_mse'] for r in v)/max(sum(r['zero_mse'] for r in v),1e-12)
    write_json(a.out,dict(records=records,pooled_recovered_energy=totals,checkpoint_sha256=file_hash(a.checkpoint),code_sha256=file_hash(__file__),crop=[480,200,800,520],interpretation='saved auxiliary linear heads only; not an information-theoretic bound'))
    print('PHASE_HEAD_RECOVERY',totals,flush=True)


if __name__=='__main__':main()
