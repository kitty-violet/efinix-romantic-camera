"""Capacity diagnostic only: optimize target-derived7-channel latents.

This is NOT source inference, training generalization, or a deployable model.
It checks whether two-side colour fields can describe sharp clean regions
through the exact existing integer decoder before training an encoder.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_two_colour_geometry import install,install_late,latent
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .runtime import configure_workspace,write_json


def decode(net,code,proxy):
    nodes={'up1_low_pw':F.pad(net.round(code).clamp(0,127),(0,0,0,0,0,9))};active=False
    for r in net.plan['nodes']:
        key=r['id'].replace('.','__');name=r['id']
        if name=='up1_low_pw':active=True;continue
        if not active:continue
        x=nodes[r['inputs'][0]]
        if key in net.weight:
            mult=getattr(net,key+'_m');s=(mult/131072).float().flatten()
            w=net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127)
            b=net.round(net.bias[key]/s).double()[None,:,None,None]
            if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
            dot=F.conv2d(x,w,stride=r['stride'],groups=r['groups'])
            z=net.round((dot.double()+b).clamp(-2**31,2**31-1)*mult/131072).clamp(-128,127).float()
            if r['activation']=='relu':z=z.clamp_min(0)
        else:z=x.repeat_interleave(2,-2).repeat_interleave(2,-1)
        if proxy and name=='up2_low_pw' and not getattr(net,'late_renderer',False):
            f=nodes['up1_high_dw'];p=((f[:,6:7]-63.5)/2).sigmoid()
            smooth=F.pad(torch.cat((f[:,:3]*(1-p),f[:,3:6]*p),1),(0,0,0,0,0,2))
            z=ExactValueProxyGradient.apply(z,smooth)
        nodes[name]=z
    return nodes['rgb_head']/128


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--steps',type=int,default=400);ap.add_argument('--late',action='store_true');a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    net=m.t.r.initialize();net.quantized=True;(install_late if a.late else install)(net);net.late_renderer=a.late;net.requires_grad_(False)
    rows={r['id']:r for r in m.load_data()['validation']};targets=[];codes=[]
    for key,box,label in m.t.r.c.CROPS:
        row=rows[key];x0,y0,x1,y1=box
        width=x1-x0;height=y1-y0;assert width%4==0 and height%4==0
        x0=x0//4*4;y0=y0//4*4;x1=x0+width;y1=y0+height
        target=m.t.r.read(row['target'])[y0:y1,x0:x1];targets.append(m.t.r.tensor(target))
        code=latent(row['target'],row['region'])[:,y0//4:y1//4,x0//4:x1//4];codes.append(torch.from_numpy(code.copy()).cuda())
    target=torch.stack(targets);code=torch.nn.Parameter(torch.stack(codes));opt=torch.optim.Adam([code],lr=.25)
    before=decode(net,code,False).detach();same=decode(net,code,True).detach();assert torch.equal(before,same)
    history=[]
    for step in range(a.steps):
        pred=decode(net,code,True);loss=(pred-target).abs().mean()
        # Correct the target directly; avoid a hand-crafted blur/noise tradeoff.
        loss+=.25*sum((torch.diff(pred,dim=k)-torch.diff(target,dim=k)).abs().mean() for k in (-1,-2))
        opt.zero_grad();loss.backward();assert torch.isfinite(code.grad).all();opt.step()
        with torch.no_grad():code.clamp_(0,127)
        if (step+1)%100==0:history.append(dict(step=step+1,loss=float(loss.detach())));print('TWO_COLOUR_LATENT',history[-1],flush=True)
    with torch.no_grad():after=decode(net,code,False)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(_,_,label) in enumerate(m.t.r.c.CROPS):
        for j,(z,title) in enumerate(zip((target,before,after),('区域目标','注入初值·非预测','拟合已知目标潜变量·非预测'))):
            im=m.t.r.c.rgb_image(z[i]);board.paste(im,(j*420,i*314+34));draw.text((j*420+3,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(history=history,
        before_mae=float((before-target).abs().mean()*128),after_mae=float((after-target).abs().mean()*128),
        exact_forward_with_and_without_proxy=True,late720_renderer=a.late,target_optimized_latents=True,source_prediction=False,deployable=False))


if __name__=='__main__':main()
