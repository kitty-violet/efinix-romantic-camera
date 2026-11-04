"""Separate target detail from artifacts of the fixed hardware color decoder."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .train_luma_regions import ROOT,RUN,LEVELS
from .runtime import file_hash,write_json


def categorical(rgb):
    x=rgb.astype(np.float32);r,g,b=np.moveaxis(x,-1,0)
    cs=[.25*r+.5*g+.25*b,.5*r-.5*b,-.25*r+.5*g-.25*b]
    y,co,cg=[np.asarray(v)[np.abs(c[...,None]-np.asarray(v)).argmin(-1)] for c,v in zip(cs,LEVELS)]
    return np.stack([y+co-cg,y+cg,y-co-cg],-1).clip(0,255).astype(np.uint8)


def main():
    root=ROOT/'runs'/RUN;out=root/'intermediate-representation-audit';out.mkdir(exist_ok=False)
    data=json.loads((root/'native-mid-fp32/data.json').read_text(encoding='utf-8'));rows=[]
    for r in data['validation']:
        path=root/'intermediate-target-screen'/f'{r["id"]}_0.008.png';t=np.asarray(Image.open(path).convert('RGB'));q=categorical(t)
        Image.fromarray(q).save(out/(r['id']+'.png'))
        rows.append(dict(id=r['id'],target_sha256=file_hash(path),quantized_rgb_mae=float(np.abs(t.astype(float)-q).mean())))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(960,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        paths=[root/'intermediate-target-screen'/f'{key}_0.008.png',out/(key+'.png'),root/'native-mid-fp32/fp32_000500'/(key+'.png')]
        for i,(path,title) in enumerate(zip(paths,['连续区域目标','硬件色表直接表示目标','当前网络 500步'])):
            d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(i*320,j*298+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='TARGET_REPRESENTATION_DIAGNOSTIC',records=rows,not_network_inference=True,not_blind=True,source_sha256=file_hash(__file__)))


if __name__=='__main__':main()
