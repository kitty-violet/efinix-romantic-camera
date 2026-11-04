"""Optimize known-target latent grids to check contour loss/decoder capacity.

Not source prediction; no claim of a trained deployable model. Existing exact
decoder values with the current ternary derivative proxies are used in both
arms. Only the new iso-contour curve term differs.
"""
import argparse,copy,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_ternary_regions import TernaryB1,region_codes,field_pre
from .train_b1_contour_coherence import ContourCoherenceTraining
from .probe_b1_ternary_field_targets import palette
from .probe_b1_decoder_coverage import fragment
from .b1_field_curve_loss import field_curve_loss,regression
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def decode(net,code,proxy=True):
    nodes={'up1_low_pw':net.round(code).clamp(0,127)};active=False
    for r in net.plan['nodes']:
        name=r['id'];key=name.replace('.','__')
        if name=='up1_low_pw':active=True;continue
        if not active:continue
        x=nodes[r['inputs'][0]]
        if key in net.weight:
            mult=getattr(net,key+'_m');s=(mult/131072).float().flatten()
            w=net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127);b=net.round(net.bias[key]/s).double()[None,:,None,None]
            if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
            raw=net.round((F.conv2d(x,w,stride=r['stride'],groups=r['groups']).double()+b).clamp(-2**31,2**31-1)*mult/131072).float()
            y=raw.clamp(-128,127)
            if r['activation']=='relu':y=y.clamp_min(0)
            if proxy:y=net.training_activation_proxy(name,y,raw,r)
        else:y=x.repeat_interleave(2,-2).repeat_interleave(2,-1)
        if proxy:y=net.training_node_proxy(name,y,nodes)
        nodes[name]=y
    return nodes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=400);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100719);torch.set_num_threads(2);regression();dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);pk=palette(net);net=net.cuda();net.__class__=TernaryB1;net.requires_grad_(False)
    byid={r['id']:r for r in m.load_data(a.data)['validation']};xs=[];ts=[];labels=[];codes=[]
    for key,box,label in m.t.r.c.CROPS:
        x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+h,x0:x0+w];row=byid[key]
        def read(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy()).float()/128-1
        xs.append(read(row['path']));ts.append(read(row['target']));labels.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)));codes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
    x,t=torch.stack(xs).cuda(),torch.stack(ts).cuda()
    with torch.no_grad():source=net(x,all_nodes=True);initial=source['up1_low_pw'];original=decode(net,initial,False)
    assert torch.equal(original['rgb_head'],source['rgb_head'])
    assert all(torch.equal(v,decode(net,initial,True)[k]) for k,v in original.items())
    tr=ContourCoherenceTraining.__new__(ContourCoherenceTraining);tr.net=net;tr.args=argparse.Namespace(classification=.5);tr.step=1;tr.rng=random.Random(4);tr.labels=torch.stack(labels).cuda();tr.class_labels=torch.stack(codes).cuda();tr.pk=pk;tr.motion_aligned_fraction=x.new_zeros(())
    pictures={'source-prediction':original['rgb_head'].detach()/128};records=[]
    for strength in (0.,2.):
        z=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([z],lr=.4);history=[]
        for i in range(1,a.steps+1):
            tr.nodes=decode(net,z);p=tr.nodes['rgb_head']/128
            loss,v=tr.additional_loss(p[...,40:-40,40:-40],t[...,40:-40,40:-40],False)
            curve=field_curve_loss(field_pre(net,tr.nodes)[...,20:-20,20:-20]);loss=loss+strength*curve
            opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(z.grad).all();opt.step()
            with torch.no_grad():z.clamp_(0,127)
            if i%100==0:
                history.append(dict(step=i,loss=float(loss.detach()),curve=float(curve.detach()),flat=float(v['region_flatness'].detach())));print('TERNARY_LATENT_CURVE',strength,history[-1],flush=True)
        with torch.no_grad():nodes=decode(net,z,False);ref=fragment(net,net.round(z),'up1_low_pw');assert all(torch.equal(v,ref[k]) for k,v in nodes.items());pictures[f'latent_curve{strength}']=nodes['rgb_head']/128
        torch.save(dict(latent=z.detach().cpu(),source_prediction=False,checkpoint_sha256=file_hash(a.checkpoint)),dest/f'latent_curve{strength}.pt');records.append(dict(strength=strength,history=history))
    board=Image.new('RGB',(1680,314*len(xs)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(_,_,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,images) in enumerate([('target',t),*pictures.items()]):
            board.paste(m.t.r.c.rgb_image(images[i]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(records=records,source_prediction=False,target_optimized_latents=True,deployable=False,decoder_fragment_equal=True,exact_proxy_forward=True))


if __name__=='__main__':main()
