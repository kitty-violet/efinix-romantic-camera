"""An unconstrained16-colour palette;16 masks can mix directly into RGB.

Eight final feature channels do NOT restrict the palette to eight rays:
three channels suffice for linear RGB mixing AFTER16-mask shape refinement.
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_categorical_palette import nearest
from .runtime import configure_workspace,write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();cv2.setNumThreads(2);cv2.setRNGSeed(2026100657);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    d=m.load_data();rows=d['training'][::4]
    points=np.concatenate([m.t.r.read(r['target'])[::12,::12].reshape(-1,3) for r in rows])
    codes=points.astype(np.int64)//4;key=codes[:,0]*4096+codes[:,1]*64+codes[:,2]
    unique,count=np.unique(key,return_counts=True);rgb=np.stack((unique//4096,(unique//64)%64,unique%64),1)*4+2
    repeats=np.maximum(1,np.rint(count.astype(float)**.7)).astype(int)
    samples=np.repeat(rgb,repeats,axis=0).astype(np.float32)
    # Fit in RGB; retain a single constant RGB triple for every selected class.
    compact,_,palette=cv2.kmeans(samples,16,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,.03),5,cv2.KMEANS_PP_CENTERS)
    order=np.argsort(palette@np.array([.25,.5,.25]));palette=palette[order].round().clip(0,255)
    byid={r['id']:r for r in d['validation']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        target=m.t.r.read(byid[key]['target'])[box[1]:box[3],box[0]:box[2]]
        q=palette[nearest(target,palette)].reshape(target.shape).astype(np.uint8)
        for j,(im,title) in enumerate(((target,'离线目标'),(q,'16个自由颜色·仅容量诊断'))):
            board.paste(Image.fromarray(im),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'target-capacity.png')
    write_json(dest/'palette.json',dict(palette=palette.tolist(),fit_compactness=float(compact),training_ids=[r['id'] for r in rows],
        data_sha256=file_hash(m.OUT/'data.json'),target_capacity_only=True,network_trained=False,visual_acceptance=False))
    shutil.copy2(__file__,dest/Path(__file__).name);print('FREE_PALETTE_COMPLETE',palette.tolist(),flush=True)


if __name__=='__main__':main()
