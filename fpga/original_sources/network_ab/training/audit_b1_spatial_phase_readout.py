"""Test whether neighbouring features expose phase hidden from a 1x1 head.

Ridge strength selected on held-out TRAIN videos, then frozen for seen-val.
This is a diagnostic readout, not an extra deployable operation.
"""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def fit(x,y,alpha):
    x=x.double();y=y.double();mu=x.mean(0);std=x.std(0).clamp_min(.001)
    a=(x-mu)/std;a=torch.cat((a,a.new_ones(len(a),1)),1)
    reg=torch.eye(a.shape[1],dtype=a.dtype)*len(a)*alpha;reg[-1,-1]=0
    return mu,std,torch.linalg.solve(a.T@a+reg,a.T@y)


def predict(x,model):
    mu,std,coef=model;a=(x.double()-mu)/std
    return torch.cat((a,a.new_ones(len(a),1)),1)@coef


def energy(p,t):
    p=p.double();t=t.double()
    return {name:1-float((p[:,j:j+16]-t[:,j:j+16]).square().sum()/t[:,j:j+16].square().sum().clamp_min(1e-12)) for name,j in (('source',0),('style',16))}


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,net=load(a.checkpoint);d=m.load_data(a.data)
    videos=sorted({r['video'] for r in d['training']});rows=[]
    for vi,video in enumerate(videos):
        rr=[r for r in d['training'] if r['video']==video]
        rows.extend((('fit' if vi<5 else 'tune'),rr[i]) for i in sorted(set((0,len(rr)//2,len(rr)-1))))
    rows.extend(('seen_validation',r) for r in d['validation'])
    samples={k:[] for k in ('fit','tune','seen_validation')};gen=torch.Generator().manual_seed(2026100744)
    for split,row in rows:
        tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
        x,t=tensor(m.t.r.read(row['path'])),tensor(m.t.r.read(row['target']));nodes=net(x,all_nodes=True);f=nodes['down2.pw']/64
        h,w=f.shape[-2:];yy=torch.randint(12,h-12,(1024,),generator=gen);xx=torch.randint(12,w-12,(1024,),generator=gen)
        point=f[0,:,yy,xx].T
        spatial=torch.cat([f[0,:,yy+dy,xx+dx].T for dy in (-1,0,1) for dx in (-1,0,1)],1)
        targets=[]
        for image in (x,t):
            y=(image[:,:1]+2*image[:,1:2]+image[:,2:3])/4;phase=F.pixel_unshuffle(y,4);phase-=phase.mean(1,keepdim=True)
            targets.append(phase[0,:,yy,xx].T)
        samples[split].append((point,spatial,torch.cat(targets,1)))
    stacked={split:[torch.cat([v[i] for v in values]) for i in range(3)] for split,values in samples.items()}
    results={}
    for index,name in ((0,'pointwise'),(1,'spatial3x3')):
        xf,_,yf=stacked['fit'];xf=stacked['fit'][index];xt=stacked['tune'][index];yt=stacked['tune'][2]
        choices=[]
        for alpha in (.001,.01,.1,1.):
            model=fit(xf,yf,alpha);score=energy(predict(xt,model),yt);choices.append(dict(alpha=alpha,held_train_energy=score))
        selected=max(choices,key=lambda v:v['held_train_energy']['source'])['alpha']
        xall=torch.cat((xf,xt));yall=torch.cat((yf,yt));model=fit(xall,yall,selected)
        results[name]=dict(alpha=selected,tuning=choices,refit_train_energy=energy(predict(xall,model),yall),seen_validation_energy=energy(predict(stacked['seen_validation'][index],model),stacked['seen_validation'][2]))
    write_json(a.out,dict(results=results,checkpoint_sha256=file_hash(a.checkpoint),code_sha256=file_hash(__file__),node='down2.pw',scale=4,rows=[dict(id=r['id'],split=s) for s,r in rows],samples_per_frame=1024,not_an_information_theory_bound=True,deployable=False))
    print('SPATIAL_PHASE_READOUT',results,flush=True)


if __name__=='__main__':main()
