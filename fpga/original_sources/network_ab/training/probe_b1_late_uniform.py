"""Retain learned sampling features; quantize before the final spatial cleanup."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .seed_b1_clear_regions import BASES,INCREMENTS
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .publish_b1_style_recovery import export


@torch.no_grad()
def convert(net,unit=8.,outline=.5):
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw']
    s=w.new_tensor(INCREMENTS)/127;offset=b.new_tensor(BASES)
    cw=torch.cat(((w[:6]*s[:,None,None,None]).mean(0,keepdim=True),w[6:].clone()),0)
    cb=torch.cat(((b[:6]*s+offset).mean()[None],b[6:].clone()))
    steps=w.new_tensor((unit,3.,3.));phase=w.new_tensor((0.,(-64)%3,(-64)%3))
    w.zero_();b.zero_();w[:3]=cw/steps[:,None,None,None];b[:3]=(cb+phase)/steps
    for c in range(3,7):w[c]=cw[0];b[c]=cb[0]
    dw,db=net.weight['up2_high_dw'],net.bias['up2_high_dw'];dw.zero_();db.zero_()
    for c in range(3):dw[c,0]=1/9
    sobel=w.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/4
    for c,k in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):dw[c,0]=k;db[c]=-1.5
    h,hb=net.weight['rgb_head'],net.bias['rgb_head'];h.zero_()
    matrix=w.new_tensor([[1,1,-1],[1,0,1],[1,-1,-1]])
    h[:,:3,0,0]=matrix*steps[None]*2;h[:,3:7,0,0]=-outline
    hb.copy_(-2*(matrix*(64+phase)[None]).sum(1))
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):
        w=net.weight[key];peak=w.abs().flatten(1).amax(1).clamp_min(1e-5)
        scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--unit',type=float,default=8.)
    ap.add_argument('--warm',default=str(t.OUT/'clear-regions/best.pt'));args=ap.parse_args()
    configure_workspace();set_reproducible(2026100683);dest=OUT/args.name;dest.mkdir(exist_ok=False)
    net=t.r.initialize();state=torch.load(args.warm,map_location='cpu',weights_only=True);net.load_state_dict(state['model']);net.quantized=True
    convert(net,args.unit);d=t.data();export(net,d['validation'],dest/'package')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,
        parent_checkpoint=args.warm,parent_training_step=state['step'],head_replacement=True),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview');shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(args=vars(args),added_operators=0,training_steps_after_conversion=0,
        minimum_area_guarantee=False,parent_sha256=file_hash(args.warm),code_sha256=file_hash(__file__)))
    print('LATE_UNIFORM_PROBE',dest,flush=True)


if __name__=='__main__':main()
