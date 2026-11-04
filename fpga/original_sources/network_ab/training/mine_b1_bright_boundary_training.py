"""TRAIN-only localization of broad bright-boundary residuals.

Do not mine validation. A small target-neighbourhood tolerance excludes tiny
alignment errors; the criterion is a diagnostic, not a semantic highlight mask.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data(a.data);records=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=sorted([r for r in data['training'] if r['video']==video],key=lambda r:r['id']);best=None
        for i in sorted(set((0,len(rows)//2,len(rows)-1))):
            row=rows[i];src=m.t.r.read(row['path']);target=m.t.r.read(row['target'])
            x=torch.from_numpy(src.transpose(2,0,1).copy())[None].float()/128-1
            t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float();p=net(x)[0]*128+128
            lum=lambda z:z[:,0:1]*.25+z[:,1:2]*.5+z[:,2:3]*.25
            yt=lum(t);yp=lum(p);edge=F.max_pool2d(yt,17,1,8)+F.max_pool2d(-yt,17,1,8)>24
            excess=(yp-F.max_pool2d(yt,5,1,2)-4).clamp_min(0)*edge*(yp>100)
            score=F.avg_pool2d(excess,192,16)[0,0];idx=int(score.argmax());iy,ix=divmod(idx,score.shape[1]);yy=min(400,max(0,iy*16-64));xx=min(960,max(0,ix*16-64))
            result=dict(row=row,xy=[xx,yy],wh=[320,320],score=float(score.flatten()[idx]))
            if best is None or result['score']>best['score']:
                best=result
                pics=[Image.fromarray(src),Image.fromarray(target),m.t.r.c.rgb_image((p[0]-128)/128)]
        records.append(best);xx,yy=best['xy'];box=(xx,yy,xx+320,yy+320)
        board=Image.new('RGB',(960,354),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
        for j,(title,im) in enumerate(zip(('TRAIN原图','当前监督目标','实际B1'),pics)):
            board.paste(im.crop(box),(j*320,34));draw.text((j*320+4,4),title,font=font,fill='white')
        name=f'train_{len(records):02d}.png';board.save(dest/name);best['review_image']=str(dest/name)
        print('BRIGHT_BOUNDARY_TRAIN',best['row']['id'],best['xy'],best['score'],flush=True)
    assert all(r['row']['id'] in {x['id'] for x in data['training']} for r in records)
    write_json(dest/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),source_sha256=file_hash(__file__),train_only=True,criterion='brightness excess beyond 5x5 target envelope, within 17px strong target transition; not semantic',used_for_training=False))


if __name__=='__main__':main()
