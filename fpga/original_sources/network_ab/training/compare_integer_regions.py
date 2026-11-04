"""Fixed palette diagnostic after integer edge-aware reference, not RTL."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .region_detail_target import ROOT,RUN
from .runtime import write_json

def quant(x,step):
    x=x.astype(np.int32);r,g,b=x.transpose(2,0,1)
    y=(r+2*g+b+2)//4;co=(r-b)//2;cg=(-r+2*g-b)//4
    y=(y+step//2)//step*step;co=(co+2)//4*4;cg=(cg+2)//4*4
    return np.stack((y+co-cg,y+cg,y-co-cg),2).clip(0,255).astype(np.uint8)

def main():
    root=ROOT/'runs'/RUN;source=root/'integer-surface-probe-v1';out=root/'integer-surface-palette-v1';out.mkdir(exist_ok=False)
    data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'));kinds=[(4,8,8),(4,12,8),(4,12,16),(8,12,16)]
    for row in data['validation']:
        for it,scale,step in kinds:
            x=np.asarray(Image.open(source/f'{row["id"]}_{it}_{scale}.png'));Image.fromarray(quant(x,step)).save(out/f'{row["id"]}_{it}_{scale}_{step}.png')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1600,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        x,y=xy;paths=[root/'detail_a'/(key+'.png')]+[out/f'{key}_{it}_{sc}_{st}.png' for it,sc,st in kinds]
        for i,(path,title) in enumerate(zip(paths,['区域目标','4轮 r8 Y8','4轮 r12 Y8','4轮 r12 Y16','8轮 r12 Y16'])):
            d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='INTEGER_ALGORITHM_ONLY_NOT_RTL',variants=kinds))

if __name__=='__main__':main()
