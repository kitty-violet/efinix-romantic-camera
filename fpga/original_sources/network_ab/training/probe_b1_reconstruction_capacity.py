"""Diagnostic-only latent fitting and single training-crop fitting.

Never a deployable/generic result: these use the known original to optimize
individual examples. They localize limitations; they do not prove an optimum.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_reconstruction as r
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible


def decoder(net,z,detach_weights=False):
    begin=False
    for row in net.plan['nodes']:
        if row['id']=='res3_add':begin=True;continue
        if not begin:continue
        key=row['id'].replace('.','__')
        if key in net.weight:
            a=F.pad(z,(1,1,1,1),mode='replicate') if row['kernel']==3 else z
            w,b=net.weight[key],net.bias[key]
            if detach_weights:w,b=w.detach(),b.detach()
            z=F.conv2d(a,w,b,stride=row['stride'],groups=row['groups']).clamp(-128,127)
            if row['activation']=='relu':z=z.clamp_min(0)
        else:
            assert row['kind']=='nearest_x2'
            z=z.repeat_interleave(2,-2).repeat_interleave(2,-1)
    return z/128


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=2500);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100563);contract_guard()
    rows,_=r.data();dest=Path(a.out);dest.mkdir(parents=True,exist_ok=False)
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=r.initialize();net.load_state_dict(parent['model']);net.quantized=False
    # Select textured TRAINING crops, never secretly fit published validation.
    candidates=[]
    for row in rows[::19]:
        for y,x in [(64,64),(320,768)]:
            arr=r.read(row['path'])[y:y+256,x:x+256]
            gray=arr.astype(np.float32).mean(2)
            detail=float(np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())
            candidates.append((detail,row,x,y))
    chosen=sorted(candidates,key=lambda v:v[0],reverse=True)[:2]
    results=[];pictures=[]
    for idx,(_,row,x0,y0) in enumerate(chosen):
        x=r.tensor(r.read(row['path'])[y0:y0+256,x0:x0+256])[None]
        net.load_state_dict(parent['model']);net.requires_grad_(False)
        with torch.no_grad():nodes=net(x,all_nodes=True);baseline=nodes['rgb_head']/128
        z=nodes['res3_add'].detach().clone().requires_grad_(True)
        assert float((decoder(net,z)-baseline).abs().max())<1e-6
        opt=torch.optim.Adam([z],lr=.1);history=[]
        for step in range(a.steps):
            p=decoder(net,z);loss=F.mse_loss(p[...,40:-40,40:-40],x[...,40:-40,40:-40])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            with torch.no_grad():z.clamp_(-128,127)
            if (step+1)%500==0:history.append(dict(step=step+1,loss=float(loss.detach())));print('CAPACITY_LATENT',idx,step+1,float(loss.detach()),flush=True)
        with torch.no_grad():latent=decoder(net,z).clone()
        # Full fixed-graph single-example fit. This is intentionally not a
        # candidate for model selection or video rendering.
        net.requires_grad_(True)
        opt=torch.optim.Adam([dict(params=list(net.weight.values()),lr=.0001),dict(params=list(net.bias.values()),lr=.003)])
        wh=[]
        for step in range(a.steps):
            p=net(x)[0];loss=F.mse_loss(p[...,40:-40,40:-40],x[...,40:-40,40:-40])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
            if (step+1)%500==0:wh.append(dict(step=step+1,loss=float(loss.detach())));print('CAPACITY_SINGLE',idx,step+1,float(loss.detach()),flush=True)
        with torch.no_grad():fitted=net(x)[0]
        outputs=[x,baseline,latent,fitted];pictures.append([r.c.rgb_image(z[0]).crop((40,40,216,216)) for z in outputs])
        results.append(dict(train_source=row,xy=[x0,y0],wh=[256,256],
            metrics={key:r.metrics(z[...,40:-40,40:-40],x[...,40:-40,40:-40]) for key,z in zip(['source','baseline','oracle_latent','single_image_fit'],outputs)},
            latent_history=history,single_fit_history=wh))
        write_json(dest/'results.json',dict(parent=a.checkpoint,parent_sha256=file_hash(a.checkpoint),records=results,
            diagnostic_only=True,deployable=False,generic_quality=False,theoretical_optimum_proven=False))
    board=Image.new('RGB',(4*240,2*212),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    titles=['训练原图','通用模型','优化潜变量（不可部署）','单图拟合（不泛化）']
    for i,ims in enumerate(pictures):
        for j,im in enumerate(ims):
            board.paste(im,(j*240,i*212+32));d.text((j*240,i*212+5),titles[j],font=font,fill='white')
    board.save(dest/'diagnostic.png')


if __name__=='__main__':main()
