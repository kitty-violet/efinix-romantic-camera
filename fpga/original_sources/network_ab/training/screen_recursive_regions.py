"""CPU fixed-point recursive edge-aware feasibility screen, not deployment.

Method reference: Gastal & Oliveira, Domain Transform (SIGGRAPH 2011),
and OpenCV modules/photo/src/npr.hpp. Independent NumPy integer equations.
No network weights, no material masks and no temporal frame blending.
"""
import json,math,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .train_luma_regions import LEVELS


def recursive(x,space,color,iterations=1):
    guide=cv2.GaussianBlur(x,(5,5),.65,borderType=cv2.BORDER_REPLICATE).astype(np.int32)
    dx=np.abs(guide[:,1:]-guide[:,:-1]).sum(2);dy=np.abs(guide[1:]-guide[:-1]).sum(2)
    z=x.astype(np.int32)*256;tables=[]
    for iteration in range(iterations):
        sigma=space*math.sqrt(3)*2**(iterations-iteration-1)/math.sqrt(4**iterations-1)
        lut=np.rint(4096*np.exp(-math.sqrt(2)/sigma*(1+space/color*np.arange(766)/255))).astype(np.int32);tables.append(lut.tolist())
        hx=lut[dx][...,None];hy=lut[dy][...,None]
        for i in range(1,z.shape[1]):z[:,i]+=((z[:,i-1]-z[:,i])*hx[:,i-1]+2048)>>12
        for i in range(z.shape[1]-2,-1,-1):z[:,i]+=((z[:,i+1]-z[:,i])*hx[:,i]+2048)>>12
        for i in range(1,z.shape[0]):z[i]+=((z[i-1]-z[i])*hy[i-1]+2048)>>12
        for i in range(z.shape[0]-2,-1,-1):z[i]+=((z[i+1]-z[i])*hy[i]+2048)>>12
    return ((z+128)>>8).clip(0,255).astype(np.uint8),tables


def palette(x):
    a=x.astype(np.int32);r,g,b=a.transpose(2,0,1)
    cs=[r+2*g+b,2*r-2*b,-r+2*g-b];v=[]
    for z,l in zip(cs,LEVELS):
        levels=np.asarray(l,dtype=np.int32);v.append(levels[np.abs(z[...,None]-levels*4).argmin(-1)])
    y,co,cg=v;return np.stack((y+co-cg,y+cg,y-co-cg),2).clip(0,255).astype(np.uint8)


def main():
    cv2.setNumThreads(2);root=ROOT/'runs'/RUN;out=root/'recursive-region-screen-v1';out.mkdir(exist_ok=False)
    data=json.loads((root/'ordinal-regions-fp32/data.json').read_text(encoding='utf-8'));records=[]
    variants=[('s16r04n1',16,.4,1),('s32r08n1',32,.8,1),('s32r08n2',32,.8,2)]
    for r in data['validation']:
        x=np.asarray(Image.open(r['source']).convert('RGB'));assert file_hash(r['source'])==r['source_sha256']
        for name,s,c,n in variants:
            start=time.perf_counter();y,luts=recursive(x,s,c,n);Image.fromarray(palette(y)).save(out/f'{r["id"]}_{name}.png')
            records.append(dict(id=r['id'],variant=name,seconds=time.perf_counter()-start,lut=luts))
        print('RECURSIVE_SCREEN',r['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1600,894),'#18202a');draw=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        paths=[root/'intermediate-target-screen'/f'{key}_0.008.png',root/'ordinal-regions-fp32/fp32_001000'/(key+'.png')]+[out/f'{key}_{name}.png' for name,*_ in variants]
        for i,(path,title) in enumerate(zip(paths,['区域目标','有序头1000步']+[v[0] for v in variants])):
            draw.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='INTEGER_ALGORITHM_SCREEN_ONLY',records=records,code_sha256=file_hash(__file__),references=['https://www.inf.ufrgs.br/~eslgastal/DomainTransform/','https://github.com/opencv/opencv/blob/4.x/modules/photo/src/npr.hpp'],fpga=False,style_accepted=False))


if __name__=='__main__':main()
