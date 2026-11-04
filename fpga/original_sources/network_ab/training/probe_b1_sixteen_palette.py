"""Capacity of16 exclusive colours in eight B1 paired colour channels."""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_categorical_palette import nearest
from .runtime import configure_workspace,write_json,file_hash


def fit(points,seed):
    cv2.setRNGSeed(seed)
    _,_,centres=cv2.kmeans(points.astype(np.float32),8,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,60,.1),1,cv2.KMEANS_PP_CENTERS)
    base=np.quantile(points,.01,axis=0).clip(0,32);w=np.vstack(((centres-base)*1.65,base))
    f=np.zeros((16,9));f[:,-1]=1;k=np.full(8,9,dtype=int)
    for j in range(8):f[2*j,j]=k[j]/16;f[2*j+1,j]=1
    code=points.astype(np.int64)//4;keys=code[:,0]*4096+code[:,1]*64+code[:,2]
    keys,count=np.unique(keys,return_counts=True);freq=count.astype(float)**.65
    rgb=np.stack((keys//4096,(keys//64)%64,keys%64),1)*4+2
    best=None
    for _ in range(90):
        p=(f@w).clip(0,255);ids=nearest(rgb,p)
        loss=float((((rgb-p[ids])**2)*freq[:,None]).sum()/(3*freq.sum()))
        if best is None or loss<best[0]:best=(loss,w.copy(),k.copy(),f.copy())
        size=np.bincount(ids,weights=freq,minlength=16)
        sums=np.stack([np.bincount(ids,weights=freq*rgb[:,c],minlength=16) for c in range(3)],1)
        xx=f.T@(f*size[:,None]);xy=f.T@sums;ridge=np.eye(9)*freq.sum()*1e-5;ridge[-1,-1]*=.01
        w=np.linalg.solve(xx+ridge,xy+ridge@w)
        # Keep the common base dark: a three-region junction may temporarily
        # have incomplete mask coverage, which must not turn into a white halo.
        w[-1]=w[-1].clip(0,32)
        for j in range(8):
            selected=ids==2*j
            if not selected.any():continue
            candidate=np.arange(3,16)
            colours=(w[-1][None]+candidate[:,None]/16*w[j][None]).clip(0,255)
            errors=(((rgb[selected,None]-colours[None])**2)*freq[selected,None,None]).sum((0,2))
            k[j]=candidate[errors.argmin()];f[2*j,j]=k[j]/16
    loss,w,k,f=best
    return dict(objective=loss,weights=w,codes=k,palette=(f@w).clip(0,255))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();cv2.setNumThreads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    d=m.load_data();rows=d['training'][::4];points=np.concatenate([m.t.r.read(r['target'])[::12,::12].reshape(-1,3) for r in rows]).astype(float)
    trials=[]
    for seed in (551,971,1281):
        result=fit(points,seed);trials.append(result);print('SIXTEEN_PALETTE',seed,result['objective'],result['codes'].tolist(),flush=True)
    result=min(trials,key=lambda r:r['objective']);byid={r['id']:r for r in d['validation']}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        target=m.t.r.read(byid[key]['target'])[box[1]:box[3],box[0]:box[2]]
        q=result['palette'][nearest(target,result['palette'])].reshape(target.shape).round().clip(0,255).astype(np.uint8)
        for j,(im,title) in enumerate(((target,'离线目标'),(q,'16色容量诊断·非网络结果'))):
            board.paste(Image.fromarray(im),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'target-capacity.png')
    serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    write_json(dest/'palette.json',dict(**serial,training_ids=[r['id'] for r in rows],data_sha256=file_hash(m.OUT/'data.json'),
        target_capacity_only=True,network_trained=False,visual_acceptance=False))
    shutil.copy2(__file__,dest/Path(__file__).name)


if __name__=='__main__':main()
