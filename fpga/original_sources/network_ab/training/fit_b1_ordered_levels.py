"""Fit ordered colour levels to train-region means; never use held-out images."""
import json
from pathlib import Path
import numpy as np
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash


def weighted_levels(x,w,k,gap):
    order=np.argsort(x);xx=x[order];ww=w[order];cdf=np.cumsum(ww)/ww.sum()
    centres=np.interp((np.arange(k)+.5)/k,cdf,xx)
    for _ in range(80):
        ids=np.abs(x[:,None]-centres[None]).argmin(1)
        counts=np.bincount(ids,weights=w,minlength=k)
        new=np.divide(np.bincount(ids,weights=w*x,minlength=k),counts,out=centres.copy(),where=counts>0)
        new.sort()
        for j in range(1,k):
            if new[j]-new[j-1]<gap:
                mid=(new[j]+new[j-1])/2;new[j-1]=mid-gap/2;new[j]=mid+gap/2
        if np.max(abs(new-centres))<1e-6:break
        centres=new
    return centres.round(3).tolist()


def main():
    configure_workspace();rows=m.load_data()['training'][::4];values=[];weights=[]
    for row in rows:
        rgb=m.t.r.read(row['target']).astype(np.float64);lab=m.load_region(row['region']).astype(np.int64);n=int(lab.max())+1
        count=np.bincount(lab.ravel(),minlength=n);valid=count>0;denom=count.clip(1)
        mean=np.stack([np.bincount(lab.ravel(),weights=rgb[...,c].ravel(),minlength=n)/denom for c in range(3)],1)[valid]
        r,g,b=mean.T;values.append(np.stack(((r+2*g+b)/4,(r-b)/2,(-r+2*g-b)/4),1))
        # Balance large areas and legitimate smaller blocks. All labels>=256px.
        weight=count[valid]**.7;weights.append(weight/weight.sum())
    x=np.concatenate(values);w=np.concatenate(weights)
    levels=[weighted_levels(x[:,i],w,k,gap) for i,(k,gap) in enumerate(((8,16),(5,8),(5,8)))]
    dest=m.OUT/'ordered-levels-from-training.json';assert not dest.exists()
    write_json(dest,dict(levels=levels,training_ids=[r['id'] for r in rows],regions=len(x),data_sha256=file_hash(m.OUT/'data.json'),
        weighting='per-frame normalized area^0.7; train regions only',network_prediction=False))
    print('TRAIN_ONLY_ORDERED_LEVELS',levels,flush=True)


if __name__=='__main__':main()
