"""Fit B1's existing8-feature RGB head to training colour distribution only.

Alternating assignment is an offline coefficient optimization. Inference is
still the existing affine RGB head, without nearest-colour search or a LUT.
"""
import argparse
import itertools
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .minimum_area_blocks import OUT
from .runtime import write_json,file_hash
from .audit_b1_palette_capacity import round_away


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--opponent-axes',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);dest=OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);model=state['model']
    rows=json.loads((OUT/'data.json').read_text(encoding='utf-8'))['training'];samples=[]
    for row in rows:
        assert file_hash(row['target'])==row['target_sha256']
        rgb=np.array(Image.open(row['target']).convert('RGB'))
        samples.append(rgb[::24,::24].reshape(-1,3))
    samples=np.concatenate(samples).astype(np.float64)-128
    # A small weighted colour histogram, from train sources only.
    q=round_away(samples/2)*2;colours,count=np.unique(q,axis=0,return_counts=True);count=count.astype(np.float64)
    idx=np.array(list(itertools.product(range(9),range(5),range(5))))
    features=np.stack([np.clip(idx[:,c]-2*g,0,2)*2 for c,groups in enumerate((4,2,2)) for g in range(groups)],1).astype(np.float64)
    scale=model['rgb_head_m'].numpy().reshape(3)/131072
    w=round_away(model['weight.rgb_head'].numpy()[:,:,0,0]/scale[:,None]).clip(-127,127)*scale[:,None]
    b=round_away(model['bias.rgb_head'].numpy()/scale)*scale
    fit=np.concatenate((w.T,b[None]),0);x=np.concatenate((features,np.ones((len(features),1))),1)
    prior=fit.copy();records=[]
    axes=np.array([[1,1,1]]*4+[[1,0,-1]]*2+[[-1,1,-1]]*2,dtype=np.float64)
    if a.opponent_axes:
        theta=np.r_[(fit[:8]*axes).sum(1)/(axes*axes).sum(1),fit[8]]
        theta[:8]=theta[:8].clip(1,32);theta_prior=theta.copy()
        design=np.zeros((len(x),3,11));design[:,:,:8]=features[:,None,:]*axes.T[None]
        design[:,:,8:]=np.eye(3)[None]
        fit[:8]=theta[:8,None]*axes;fit[8]=theta[8:]
    for step in range(31):
        palette=(x@fit).clip(-128,127)
        distance=((colours[:,None]-palette[None])**2).sum(2);ids=distance.argmin(1)
        mse=float((distance[np.arange(len(ids)),ids]*count).sum()/(3*count.sum()))
        records.append(dict(step=step,mse=mse));print('TARGET_PALETTE',step,round(mse,5),flush=True)
        if step==30:break
        n=np.bincount(ids,weights=count,minlength=len(x));sums=np.stack([np.bincount(ids,weights=count*colours[:,c],minlength=len(x)) for c in range(3)],1)
        if a.opponent_axes:
            matrix=design.reshape(-1,11);weights=np.repeat(n,3)
            xx=matrix.T@(weights[:,None]*matrix);xy=matrix.T@sums.reshape(-1)
            ridge=np.eye(11)*count.sum()*.0003;ridge[8:,8:]=0
            proposed=np.linalg.solve(xx+ridge,xy+ridge@theta_prior);proposed[:8]=proposed[:8].clip(1,32)
            theta=.6*proposed+.4*theta;fit[:8]=theta[:8,None]*axes;fit[8]=theta[8:]
        else:
            xx=x.T@(n[:,None]*x);xy=x.T@sums
            ridge=np.eye(9)*count.sum()*.0003;ridge[-1,-1]=0
            proposed=np.linalg.solve(xx+ridge,xy+ridge@prior)
            # Damping limits changes; only opponent_axes explicitly preserves
            # monotone luminance and fixed opponent-colour directions.
            fit=.6*proposed+.4*fit
    oldmse=records[0]['mse'];assert records[-1]['mse']<oldmse
    w=fit[:8].T.astype(np.float32);b=fit[8].astype(np.float32)
    scale=np.maximum(np.abs(w).max(1),1e-5)/112
    multiplier=np.ceil(scale*131072).clip(1,131071)
    model['weight.rgb_head']=torch.tensor(w[:,:,None,None]);model['bias.rgb_head']=torch.tensor(b)
    model['rgb_head_m']=torch.tensor(multiplier,dtype=torch.float64)[None,:,None,None]
    # Separate checkpoint only; learned source-region mapping is not changed.
    state['parent']=a.warm;state['target_palette_fit']=True;torch.save(state,dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),training_ids=[r['id'] for r in rows],
        sampled_pixels=len(samples),histogram_colours=len(colours),history=records,fit=fit.tolist(),opponent_axes=a.opponent_axes,
        inference_nearest_colour=False,added_operators=0,visual_acceptance=False,numeric_export_verification_pending=True))


if __name__=='__main__':main()
