"""Detail-preserving region targets. Offline teacher only; no inference FFT."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .region_mechanism_probe import tensor, rgb, gaussian, l0_surface, outline
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible

ROOT=Path('E:/fpga/model-training-workspace')
RUN='region-detail-deploy-20261004'
VARIANTS={
    'detail_a':dict(sigma=.65,gamma=.65,lam=.004,ink=.4),
    'detail_b':dict(sigma=.50,gamma=.65,lam=.0025,ink=.35),
}


@torch.no_grad()
def target(x,kind='detail_a'):
    p=VARIANTS[kind]
    # A fixed monotone transform protects low-light contrasts before L0.
    # It uses no per-image statistics and no semantic material/face masks.
    z=gaussian(x,p['sigma']).clamp_min(0).pow(p['gamma'])
    z=l0_surface(z,p['lam']).clamp(0,1).pow(1/p['gamma'])
    return outline(z,strength=p['ink'])


def manifest():
    return json.loads((ROOT/'runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))


def screen(run):
    run.mkdir(exist_ok=False);m=manifest();records=[]
    for kind in VARIANTS:
        out=run/kind;out.mkdir()
        for row in m['validation']:
            start=time.perf_counter();a=np.asarray(Image.open(row['source']).convert('RGB'))
            v=rgb(target(tensor(a),kind));Image.fromarray(v).save(out/(row['id']+'.png'))
            records.append(dict(kind=kind,id=row['id'],seconds=time.perf_counter()-start))
        print('DETAIL_SCREEN',kind,flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for name,key,xy in [('stairs','val_stairs_090',(150,430)),('pillar','val_stairs_090',(424,320)),('people','val_plaza_090',(80,400))]:
        row=next(r for r in m['validation'] if r['id']==key);x,y=xy
        paths=[row['source'],ROOT/'runs/region-mechanisms-20261004/algorithm'/(key+'.png'),run/'detail_a'/(key+'.png'),run/'detail_b'/(key+'.png')]
        board=Image.new('RGB',(1280,298),'#17202a');d=ImageDraw.Draw(board)
        for i,(p,title) in enumerate(zip(paths,['原片','上版区域算法','细节A','细节B'])):
            d.text((320*i+5,5),title,font=font,fill='white');board.paste(Image.open(p).crop((x,y,x+320,y+264)),(320*i,30))
        board.save(run/(name+'_targets.png'))
    write_json(run/'target_screen.json',dict(variants=VARIANTS,records=records,code_sha256=file_hash(__file__),
        target_is_offline=True,not_style_acceptance=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['screen']);a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100420);cv2.setNumThreads(2)
    screen(ROOT/'runs'/RUN)

if __name__=='__main__':main()
