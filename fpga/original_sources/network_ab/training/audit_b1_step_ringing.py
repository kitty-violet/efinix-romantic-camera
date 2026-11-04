"""Measure new edge halos against the model's own constant-colour endpoints.

Synthetic greys and hand-selected colour pairs isolate spatial responses. They
are not semantic/style validation. Floating variants are diagnostics only.
"""
import argparse, copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_packed720 import pre720
from . import train_b1_minimum_area as m
from .runtime import write_json, file_hash


def test_inputs():
    pairs = [([a]*3, [b]*3) for a,b in ((16,80),(32,128),(64,192),
             (96,224),(16,224),(96,144),(128,160),(176,224))]
    pairs += [([30,55,20],[110,135,60]),([70,80,100],[150,165,185]),
              ([30,50,45],[205,205,195]),([45,45,50],[150,100,65])]
    yy,xx = torch.meshgrid(torch.arange(96),torch.arange(192),indexing='ij')
    d=xx-96-.375*(yy-48)
    rows=[]; records=[]
    for i,(a,b) in enumerate(pairs):
        a=torch.tensor(a).float()[:,None,None];b=torch.tensor(b).float()[:,None,None]
        for width in (0,4,12):
            alpha=(d>0).float() if width==0 else (d/width+.5).clamp(0,1)
            rows.append((a+(b-a)*alpha).round())
            records.append(dict(pair=i,source_transition_width=width,
                                left=a.flatten().tolist(),right=b.flatten().tolist()))
    return torch.stack(rows), records


def metrics(p,left,right):
    y=lambda z:.25*z[:,0]+.5*z[:,1]+.25*z[:,2]
    lo=torch.minimum(y(left),y(right));hi=torch.maximum(y(left),y(right))
    yy=y(p);high=(yy-hi[:,None,None]).clamp_min(0)
    low=(lo[:,None,None]-yy).clamp_min(0)
    v=right-left
    alpha=((p-left[:,:,None,None])*v[:,:,None,None]).sum(1)/v.square().sum(1).clamp_min(1)[:,None,None]
    nearest=left[:,:,None,None]+alpha.clamp(0,1)[:,None]*v[:,:,None,None]
    off=(p-nearest).square().mean(1).sqrt()
    return [dict(white_overshoot_max=float(h.max()),white_overshoot_fraction=float((h>8).float().mean()),
                 dark_undershoot_max=float(l.max()),off_segment_p95=float(torch.quantile(o,.95)))
            for h,l,o in zip(high,low,off)]


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);dest=Path(args.out);dest.mkdir(exist_ok=False)
    _,net=load(args.checkpoint);actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
    floating=copy.deepcopy(net);floating.quantized=False
    for k,w in floating.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()
        w.copy_(net.round(w/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None])
        floating.bias[k].copy_(net.round(floating.bias[k]/s)*s)
    src,records=test_inputs();pictures=[]
    for start in range(0,len(src),3):
        x=src[start:start+3]/128-1;nodes=net(x,all_nodes=True)
        byte_nodes,_=actual(src[start:start+3]-128,True)
        assert all(torch.equal(v,byte_nodes[k]) for k,v in nodes.items())
        ff=floating(x,all_nodes=True)
        variants={'integer':nodes['rgb_head']+128,
            'encoder_rounding_off':fragment(net,net.round(ff['up1_low_pw']),'up1_low_pw')['rgb_head']+128,
            'final_count_rounding_off':fragment(net,pre720(net,nodes),'up2_high_dw')['rgb_head']+128}
        ends=torch.stack([src[start, :,0,0],src[start,:,0,-1]])[:,:,None,None].expand(2,3,48,48)
        endpoint=actual(ends-128)[:,:,24,24]+128
        for name,p in variants.items():
            for j,metric in enumerate(metrics(p[...,16:-16,32:-32],endpoint[0:1].expand(3,-1),endpoint[1:2].expand(3,-1))):
                records[start+j].setdefault('variants',{})[name]=metric
        pic=Image.new('RGB',(192*4,96*3+24),'#17202a');draw=ImageDraw.Draw(pic)
        for col,(name,p) in enumerate([('input',src[start:start+3])]+list(variants.items())):
            draw.text((col*192,3),name,fill='white')
            for j,a in enumerate(p):
                pic.paste(Image.fromarray(a.permute(1,2,0).numpy().clip(0,255).astype(np.uint8)),(col*192,24+j*96))
        pic.save(dest/f'pair_{start//3:02d}.png')
        print('STEP_RINGING',start//3,records[start]['variants'],flush=True)
    means={name:{k:float(np.mean([r['variants'][name][k] for r in records]))
                 for k in records[0]['variants'][name]} for name in variants}
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(args.checkpoint),actual_all_node_error=0,
        records=records,means=means,scope='Synthetic endpoint diagnostic, not image/style/video acceptance'))


if __name__=='__main__':main()
