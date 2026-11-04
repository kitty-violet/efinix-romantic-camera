"""Measure label motion consistency using the existing source-only flow mask.

No candidate inference or parameter update. Compare source, pre-area style,
old minimum-area labels and current hierarchical labels on exactly the same
21 TRAIN correspondences. This audit cannot prove flow is perfect.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_distillation import read_flow
from .coherent_shared_model import warp
from .runtime import write_json,file_hash


def tensor(path):
    return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(2);dest=Path(args.out);dest.mkdir(exist_ok=False)
    manifest=Path(args.data);current=json.loads(manifest.read_text(encoding='utf-8'))
    others={}
    for name,path in [('before_area',m.t.OUT/'data.json'),('old_area',m.OUT/'data.json')]:
        dd=json.loads(path.read_text(encoding='utf-8'));others[name]={p['current']:p for p in dd['pairs']}
    records=[];images=[]
    for pair in current['pairs']:
        assert pair['split']=='train';assert file_hash(pair['flow'])==pair['flow_sha256']
        a,b=tensor(pair['current']),tensor(pair['next']);flow,valid=read_flow(pair['flow'])
        f=torch.from_numpy(flow).permute(2,0,1)[None];v=torch.from_numpy(valid.copy())[None,None]
        v[...,:40,:]=0;v[...,-40:,:]=0;v[...,:,:40]=0;v[...,:,-40:]=0
        src=(b-warp(a,f)).abs().mean(1,keepdim=True);confidence=v*torch.exp(-(src/.08)**2)
        row=dict(source=pair['source'],current=pair['current'],next=pair['next'],valid_pixels=float(confidence.sum()),stages={})
        picture=[]
        for name,p in [('source',None),('before_area',others['before_area'][pair['current']]),
                       ('old_area',others['old_area'][pair['current']]),('hierarchical',pair)]:
            aa,bb=(a,b) if p is None else (tensor(p['current_target']),tensor(p['next_target']))
            warped=warp(aa,f);delta=(bb-warped).abs().mean(1,keepdim=True)*128
            denominator=confidence.sum().clamp_min(1)
            row['stages'][name]=dict(mae=float((delta*confidence).sum()/denominator),
                over8=float(((delta>8)*confidence).sum()/denominator),over16=float(((delta>16)*confidence).sum()/denominator))
            if name=='hierarchical':
                # Source reliable but label disagrees by >16 RGB levels.
                mask=(confidence>.8)&(delta>16)
                marked=bb.clone();marked[:,0]=torch.where(mask[:,0],torch.ones_like(bb[:,0]),bb[:,0])
                marked[:,1:]=torch.where(mask,torch.full_like(bb[:,1:],-1),bb[:,1:])
                picture=[m.t.r.c.rgb_image(z[0]) for z in (b,warped,bb,marked)]
        images.append((row['stages']['hierarchical']['mae'],Path(pair['current']).stem,picture));records.append(row)
    means={name:{k:float(np.mean([r['stages'][name][k] for r in records])) for k in ('mae','over8','over16')}
           for name in ('source','before_area','old_area','hierarchical')}
    write_json(dest/'review.json',dict(records=records,mean=means,source_only_flow_confidence=True,
        data_sha256=file_hash(manifest),model_training=False,scope='TRAIN label diagnostic; imperfect flow remains possible'))
    board=Image.new('RGB',(1280,394*3),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(_,name,pics) in enumerate(sorted(images,reverse=True,key=lambda x:x[0])[:3]):
        for j,(title,img) in enumerate(zip(('source next','warped previous target','next target','>16 label disagreement'),pics)):
            # Centre crop at 1:1, not an artificially blurred thumbnail.
            board.paste(img.crop((480,180,800,540)),(320*j,394*i+34))
            draw.text((320*j+3,394*i+4),name[-24:]+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');print('TARGET_MOTION_AUDIT',means,flush=True)


if __name__=='__main__':main()
