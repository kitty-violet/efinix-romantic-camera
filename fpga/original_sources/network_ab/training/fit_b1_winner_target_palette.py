"""Train-target-only colour dictionary using actual reachable2..14 states."""
import argparse
import itertools
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_categorical_palette import nearest
from .runtime import configure_workspace,write_json,file_hash


def fit(points,seed):
    cv2.setRNGSeed(seed)
    _,_,centres=cv2.kmeans(points.astype(np.float32),8,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,50,.1),1,cv2.KMEANS_PP_CENTERS)
    features=np.zeros((8*13,9));features[:,-1]=1
    for j in range(8):features[j*13:(j+1)*13,j]=np.arange(2,15)/16
    base=np.quantile(points,.02,axis=0);weights=np.vstack(((centres-base)*1.5,base))
    code=points.astype(np.int64)//4;packed=code[:,0]*4096+code[:,1]*64+code[:,2]
    values,counts=np.unique(packed,return_counts=True);freq=counts.astype(float)**.65
    colours=np.stack((values//4096,(values//64)%64,values%64),1)*4+2
    best=None
    for _ in range(60):
        pal=(features@weights).clip(0,255);ids=nearest(colours,pal)
        objective=float(((colours-pal[ids])**2*freq[:,None]).sum()/(3*freq.sum()))
        if best is None or objective<best[0]:best=objective,weights.copy()
        size=np.bincount(ids,weights=freq,minlength=len(features))
        sums=np.stack([np.bincount(ids,weights=freq*colours[:,c],minlength=len(features)) for c in range(3)],1)
        xx=features.T@(features*size[:,None]);xy=features.T@sums
        ridge=np.eye(9)*freq.sum()*1e-5;ridge[-1,-1]*=.01
        weights=np.linalg.solve(xx+ridge,xy+ridge@weights)
    return dict(objective=best[0],weights=best[1],palette=(features@best[1]).clip(0,255),features=features)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();cv2.setNumThreads(2);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    d=m.load_data();rows=d['training'][::4];points=np.concatenate([m.t.r.read(r['target'])[::12,::12].reshape(-1,3) for r in rows]).astype(float)
    trials=[]
    for seed in (313,571,831):
        result=fit(points,seed);trials.append(result);print('WINNER_TARGET_PALETTE_FIT',seed,result['objective'],flush=True)
    result=min(trials,key=lambda z:z['objective']);state=torch.load(a.warm,map_location='cpu',weights_only=True)
    model=state['model'];oldw=model['weight.rgb_head'][:,:,0,0].numpy();oldb=model['bias.rgb_head'].numpy()+128
    old=np.stack([oldb[None]+np.arange(2,15)[:,None]*oldw[:,j][None] for j in range(8)])
    new=result['palette'].reshape(8,13,3)
    cost=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            dist=((old[i,:,None]-new[j,None])**2).sum(2)
            cost[i,j]=(dist.min(0).mean()+dist.min(1).mean())/2
    assignment=min(itertools.permutations(range(8)),key=lambda p:cost[np.arange(8),p].sum())
    weights=result['weights'].copy();weights[:8]=weights[list(assignment)]
    palette=(result['features']@weights).clip(0,255)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    byid={r['id']:r for r in d['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        target=m.t.r.read(byid[key]['target'])[box[1]:box[3],box[0]:box[2]]
        q=palette[nearest(target,palette)].reshape(target.shape).round().clip(0,255).astype(np.uint8)
        for j,(im,title) in enumerate(((target,'离线目标'),(q,'104色容量检查·非网络输出'))):
            board.paste(Image.fromarray(im),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'target-capacity.png')
    write_json(dest/'palette.json',dict(weights=weights.tolist(),palette=palette.tolist(),class_assignment=list(assignment),
        objective=result['objective'],training_ids=[r['id'] for r in rows],data_sha256=file_hash(m.OUT/'data.json'),
        target_capacity_only=True,visual_acceptance=False))
    shutil.copy2(__file__,dest/Path(__file__).name);print('WINNER_TARGET_PALETTE_COMPLETE',flush=True)


if __name__=='__main__':main()
