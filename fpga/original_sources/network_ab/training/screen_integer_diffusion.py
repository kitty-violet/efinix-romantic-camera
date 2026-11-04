"""Bounded integer edge-stopping diffusion screen, not deployable weights.

Inspired by Perona/Malik's local nonlinear diffusion, with our own triangular
piecewise-linear flux (not a reproduction of their original conductance).
Fixed Q4, four neighbors, shift/add/compare only; no learned or temporal state.
"""
import json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .train_luma_regions import ROOT,RUN,LEVELS
from .runtime import file_hash,write_json


def divide_round(x,d):return np.sign(x)*((np.abs(x)+d//2)//d)


def diffuse(a,threshold,steps):
    s=a.astype(np.int32)*16;t=threshold*16
    for _ in range(steps):
        p=np.pad(s,((1,1),(1,1),(0,0)),mode='edge');acc=np.zeros_like(s)
        for v in (p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:]):
            d=v-s;mag=np.abs(d);flux=np.maximum(0,np.minimum(mag,2*t-mag));acc+=np.sign(d)*flux
        s=s+divide_round(acc,4)
        assert s.min()>=0 and s.max()<=255*16
    return (s+8)//16


def palette(a):
    r,g,b=[v.astype(np.int32) for v in a.transpose(2,0,1)];comp=[divide_round(r+2*g+b,4),divide_round(r-b,2),divide_round(-r+2*g-b,4)]
    q=[np.asarray(lev)[np.abs(c[...,None]-np.asarray(lev)).argmin(-1)] for c,lev in zip(comp,LEVELS)];y,co,cg=q
    return np.stack((y+co-cg,y+cg,y-co-cg),-1).clip(0,255).astype(np.uint8)


def main():
    cv2.setNumThreads(2);out=ROOT/'runs'/RUN/'integer-diffusion-screen-v1';out.mkdir(exist_ok=False);data=json.loads((ROOT/'runs'/RUN/'component-regions-fp32/data.json').read_text(encoding='utf-8'))
    trials=[(12,16),(24,16),(32,16),(24,32)];checks=[];records=[]
    for t,n in trials:
        a=np.full((32,48,3),93,np.uint8);assert np.array_equal(diffuse(a,t,n),a)
        a[:,:24]=30;a[:,24:]=220;v=diffuse(a,t,n);assert np.array_equal(v,a)
        checks.append(dict(threshold=t,steps=n,constant_and_step_exact=True))
        for r in data['validation']:
            arr=np.asarray(Image.open(r['source']).convert('RGB'));start=time.perf_counter();v=palette(diffuse(arr,t,n));Image.fromarray(v).save(out/f'{r["id"]}_t{t}_n{n}.png');records.append(dict(id=r['id'],t=t,n=n,seconds=time.perf_counter()-start));print('DIFFUSION_SCREEN',r['id'],t,n,flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1920,894),'#18202a');draw=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        r=next(r for r in data['validation'] if r['id']==key);paths=[(r['source'],'原片'),(ROOT/'runs'/RUN/'intermediate-target-screen'/f'{key}_0.008.png','区域目标')]+[(out/f'{key}_t{t}_n{n}.png',f'整数局部扩散 T{t} ×{n}') for t,n in trials]
        for i,(path,label) in enumerate(paths):draw.text((320*i+5,298*j+5),label,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='INTEGER_ALGORITHM_SCREEN_ONLY',checks=checks,records=records,source_sha256=file_hash(__file__),reference='https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf',custom_triangular_flux=True,iterations_have_hardware_cost=True,rtl=False,fit=False,timing=False,style_accepted=False))


if __name__=='__main__':main()
