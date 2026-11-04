"""Separate upstream spatial transition spread from the final RGB filter.

Same B1 coefficients/graph. Integer kernel sums are preserved per channel.
This can still change nonlinear image behaviour; no quality guarantee.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .audit_b1_rim_tolerance import rim_terms
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def concentrate(net,key,fraction):
    scale=(getattr(net,key+'_m')/131072).float().flatten()
    w=net.round(net.weight[key]/scale[:,None,None,None]).clamp(-127,127)
    original=w.flatten(1).sum(1).clone();moves=[]
    for c in range(len(w)):
        v=w[c,0].flatten();room=int(127-v[4]);count=0
        for j in sorted((i for i in range(9) if i!=4),key=lambda i:-float(v[i])):
            n=min(room,max(0,int(round(float(v[j])*fraction))));v[j]-=n;v[4]+=n;room-=n;count+=n
        moves.append(count)
    assert torch.equal(w.flatten(1).sum(1),original) and w.abs().max()<=127
    net.weight[key].copy_(w*scale[:,None,None,None]);return moves


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    models={'original':base};moves={}
    configs=[('first25',('up1_high_dw',),.25),('first50',('up1_high_dw',),.5),('middle50',('up2_low_dw',),.5)]
    for name,keys,fraction in configs:
        n=copy.deepcopy(base);moves[name]={k:concentrate(n,k,fraction) for k in keys};models[name]=n
    ims={};results={k:[] for k in models}
    for row in m.load_data(a.data)['validation']:
        tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
        x,t=tensor(m.t.r.read(row['path'])),tensor(m.t.r.read(row['target']))
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None];ims[row['id']]=[m.t.r.c.rgb_image(t[0])]
        for name,net in models.items():
            nodes=net(x,all_nodes=True);raw=net.export();assert len(raw)==20480
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            p=actual['rgb_head']/128;rim,_=rim_terms(p,t,labels,2);profile,_=complete_rgb_profiles(p,t,labels)
            results[name].append(dict(id=row['id'],mae=float((p-t).abs().mean()*128),rim=float(rim),profile=float(profile),all_nodes_error=0))
            ims[row['id']].append(m.t.r.c.rgb_image(p[0]))
        print('PREDECISION_KERNEL',row['id'],flush=True)
    titles=('目标','实际原系数','第一上采样核集中1/4','第一上采样核集中1/2','第二低分辨率核集中1/2')
    board=Image.new('RGB',(420*len(titles),314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(ims[rid],titles)):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png');Path(out/'executed_probe.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'review.json',dict(results=results,moves=moves,kernel_DC_unchanged=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),diagnostic_only=True,visual_acceptance=False))


if __name__=='__main__':main()
