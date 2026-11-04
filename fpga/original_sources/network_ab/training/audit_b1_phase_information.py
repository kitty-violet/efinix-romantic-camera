"""Diagnostic linear recovery of within-cell luminance sampling phases.

Fit on TRAIN, report on validation. This tests information accessible to a
simple readout, not an information-theoretic bound or a deployable extra head.
"""
import argparse
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);d=m.load_data(a.data);train=[]
    for video in sorted({r['video'] for r in d['training']}):
        rows=[r for r in d['training'] if r['video']==video]
        train.extend(rows[i] for i in sorted(set((0,len(rows)//2,len(rows)-1))))
    cases=[('down1.pw',2),('down2.pw',4),('res3_add',4)]
    results=[]
    for checkpoint in a.checkpoints:
        _,net=load(checkpoint);samples={k:[[],[],[]] for k,_ in cases}
        for row in train:
            source=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(source,all_nodes=True)
            for key,scale in cases:
                f=nodes[key][...,10:-10:3,10:-10:3].permute(0,2,3,1).reshape(-1,nodes[key].shape[1]).double()/64
                samples[key][0].append(f)
                for j,z in enumerate((source,target),1):
                    y=(z[:,:1]+2*z[:,1:2]+z[:,2:3])/4
                    phases=F.pixel_unshuffle(y,scale);phases-=phases.mean(1,keepdim=True)
                    samples[key][j].append(phases[...,10:-10:3,10:-10:3].permute(0,2,3,1).reshape(-1,scale*scale).double())
        fits={};checks={}
        for key,scale in cases:
            x=torch.cat(samples[key][0]);mu=x.mean(0);std=x.std(0).clamp_min(.001);x=(x-mu)/std;x=torch.cat((x,x.new_ones(len(x),1)),1)
            reg=torch.eye(x.shape[1],dtype=x.dtype)*len(x)*1e-4;reg[-1,-1]=0
            fits[key]=dict(mu=mu,std=std,coef=[])
            for j in (1,2):fits[key]['coef'].append(torch.linalg.solve(x.T@x+reg,x.T@torch.cat(samples[key][j])))
            checks[key]=[]
        for row in d['validation']:
            source=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(source,all_nodes=True)
            for key,scale in cases:
                fit=fits[key];x=nodes[key][...,10:-10:3,10:-10:3].permute(0,2,3,1).reshape(-1,nodes[key].shape[1]).double()/64
                x=(x-fit['mu'])/fit['std'];x=torch.cat((x,x.new_ones(len(x),1)),1);rec=dict(id=row['id'])
                for j,(name,z) in enumerate((('source',source),('style_target',target))):
                    y=(z[:,:1]+2*z[:,1:2]+z[:,2:3])/4
                    phase=F.pixel_unshuffle(y,scale);phase-=phase.mean(1,keepdim=True)
                    phase=phase[...,10:-10:3,10:-10:3].permute(0,2,3,1).reshape(-1,scale*scale).double();p=x@fit['coef'][j]
                    rec[name]=dict(phase_mae_byte=float((p-phase).abs().mean()*128),zero_phase_mae_byte=float(phase.abs().mean()*128),
                        residual_energy_fraction=float((p-phase).square().sum()/phase.square().sum().clamp_min(1e-12)))
                checks[key].append(rec)
        results.append(dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),nodes=checks))
        write_json(a.out,dict(results=results,training_ids=[r['id'] for r in train],readout='TRAIN fit ridge pointwise, validation only evaluation',
            purpose='diagnostic, not deployable extra head; high residual is not proof of information loss',source_sha256=file_hash(__file__)))
        print('PHASE_INFORMATION_COMPLETE',checkpoint,flush=True)


if __name__=='__main__':main()
