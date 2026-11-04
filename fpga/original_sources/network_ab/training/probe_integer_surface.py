"""Independent fixed-integer edge-aware surface reference; no GPU needed.

This is a measured alternative to learned texture removal, not yet RTL.
Range weights are powers of two, normalization is an explicit integer divide.
"""
import argparse,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .region_detail_target import ROOT,RUN
from .runtime import write_json,file_hash


def axis_filter(x,axis,scale):
    h,w,_=x.shape;pad=((0,0),(3,3),(0,0)) if axis==1 else ((3,3),(0,0),(0,0))
    padded=np.pad(x,pad,mode='edge');value=np.zeros_like(x,dtype=np.int32);denom=np.zeros((h,w),np.int32)
    for d,spatial in zip(range(-3,4),[1,2,4,8,4,2,1]):
        q=padded[:,d+3:d+3+w] if axis==1 else padded[d+3:d+3+h]
        dist=np.max(np.abs(q.astype(np.int16)-x.astype(np.int16)),2)
        shift=np.minimum(5,dist//scale);weight=(32>>shift).astype(np.int32)*spatial
        # Stop diffusion across large object/color boundaries.
        weight[dist>=5*scale]=0
        value+=q.astype(np.int32)*weight[:,:,None];denom+=weight
    return ((value+denom[:,:,None]//2)//denom[:,:,None]).astype(np.uint8)


def apply(x,iterations,scale):
    for i in range(iterations):
        for axis in ((1,0) if i%2==0 else (0,1)):x=axis_filter(x,axis,scale)
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='integer-surface-probe-v1');a=p.parse_args()
    root=ROOT/'runs'/RUN;out=root/a.name;out.mkdir(exist_ok=False);data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'));records=[]
    variants=[(4,8),(4,12),(4,16),(8,12)]
    for row in data['validation']:
        x=np.asarray(Image.open(row['source']).convert('RGB'))
        for it,sc in variants:
            start=time.perf_counter();y=apply(x,it,sc);Image.fromarray(y).save(out/f'{row["id"]}_{it}_{sc}.png')
            records.append(dict(id=row['id'],iterations=it,scale=sc,seconds=time.perf_counter()-start))
        print('INTEGER_SURFACE',row['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1600,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        x,y=xy;paths=[root/'detail_a'/(key+'.png')]+[out/f'{key}_{it}_{sc}.png' for it,sc in variants]
        for i,(path,title) in enumerate(zip(paths,['区域目标','4轮 range8','4轮 range12','4轮 range16','8轮 range12'])):
            d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='INTEGER_ALGORITHM_ONLY_NOT_RTL',code_sha256=file_hash(__file__),records=records))

if __name__=='__main__':main()
