"""CPU control experiment: spatial colour choice, colour separation, outlines.

All controls are encoded as ordinary coefficients in the original B1 layers.
The outline derives from quantized region brightness, never raw input texture.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .train_b1_learned_colour import ColourB1,CROPS
from .probe_b1_cartoon_int8 import IntegerB1
from .b1_trained_image import runtime_from_image
from .runtime import file_hash,write_json


@torch.no_grad()
def modify(net,smooth=False,outline=0.,tone=False,majority=False,sharpen=0.):
    if sharpen:
        # Sharpen the learned low-resolution features BEFORE colour decisions.
        # Preserve each kernel's DC gain; never sharpen the finished RGB image.
        key='up1_low_dw';w=net.weight[key]
        w.copy_((1+sharpen)*w-sharpen*w.mean((-1,-2),keepdim=True))
        mult=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/120).ceil().clamp(1,131071)
        net.up1_low_dw_m.copy_(mult.double()[None,:,None,None])
    if majority:
        # Make region membership decisive before two existing spatial votes.
        # Voting operates on region decisions, never averages displayed RGB.
        key='up1_low_pw';net.weight[key][:15]*=8
        net.bias[key][:15]=(net.bias[key][:15]-63.5)*8+63.5
        w=net.weight[key];mult=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/120).ceil().clamp(1,131071)
        net.up1_low_pw_m.copy_(mult.double()[None,:,None,None])
        scale=131071/131072
        net.weight['up1_high_dw'][:15]=127*scale
        net.bias['up1_high_dw'][:15]=-127*(9*64-1)*scale
        net.up1_high_dw_m[:,:15]=131071
        smooth=True
    if smooth:
        scale=131071/131072
        net.weight['up2_low_dw'][:15]=127*scale
        net.bias['up2_low_dw'][:15]=-127*(9*64-1)*scale
        net.up2_low_dw_m[:,:15]=131071
    if tone:
        w,b=net.weight['rgb_head'],net.bias['rgb_head']
        y=w.new_tensor([.299,.587,.114])[None].expand(3,-1)
        transform=1.10*(y+1.15*(torch.eye(3,device=w.device)-y))
        w.copy_(torch.einsum('ij,jchw->ichw',transform,w.clone()))
        b.copy_(transform@b.clone()+b.new_tensor([-5,0,7]))
    if outline:
        w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw']
        w[3:7]=w[:1];b[3:7]=b[:1]
        net.up2_low_pw_m[:,3:7]=net.up2_low_pw_m[:,:1]
        dw,db=net.weight['up2_high_dw'],net.bias['up2_high_dw']
        sobel=dw.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/4
        for c,k in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):dw[c,0]=k;db[c]=-4
        net.up2_high_dw_m[:,3:7]=1024
        net.weight['rgb_head'][:,3:7,0,0]=-outline
    if tone or outline:
        w=net.weight['rgb_head'];mult=(w.abs().flatten(1).amax(1)*131072/120).ceil().clamp(1,131071)
        net.rgb_head_m.copy_(mult.double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--majority',action='store_true');ap.add_argument('--sharpen',type=float,default=0.);a=ap.parse_args()
    torch.set_num_threads(2);root=Path(__file__).resolve().parents[2]/'model-training-workspace/runs'
    data=json.loads((root/'b1-minimum-area-20261006/data.json').read_text(encoding='utf-8'))
    state=torch.load(a.warm,map_location='cpu',weights_only=True)
    base=ColourB1(revive=False,source=root/'b1-identity-reconstruction-20261005/initial_identity.bin')
    base.load_state_dict(state['model']);base.package['residual_encoding']=state['residual_encoding'];base.quantized=True
    dest=Path(a.out);dest.mkdir(exist_ok=False);images={};records=[]
    variants=[('current',False,0,False),('choice-smooth',True,0,False),('colour-outline',True,.75,True)]
    if a.majority:variants=[('majority',True,.75,True)]
    for label,smooth,outline,tone in variants:
        net=modify(copy.deepcopy(base),smooth,outline,tone,a.majority,a.sharpen);sub=dest/label;sub.mkdir();raw=net.export()
        (sub/'coefficients-padded.bin').write_bytes(raw)
        actual=IntegerB1(runtime_from_image(raw),device='cpu');records_sub=[]
        for row in data['validation']:
            rgb=np.array(Image.open(row['path']).convert('RGB'));x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);masks=nodes['up2_low_dw'][:,:15]
            assert torch.all((masks==0)|(masks==127))
            for start,end in ((0,7),(7,11),(11,15)):assert torch.all(masks[:,start:end-1]>=masks[:,start+1:end])
            im=Image.fromarray((nodes['rgb_head'][0].permute(1,2,0).numpy()+128).astype(np.uint8))
            im.save(sub/(row['id']+'.png'));images[(label,row['id'])]=im
            records_sub.append(dict(id=row['id'],binary_ordered_decisions=True))
            del nodes
        child=dict(state);child.update(model=net.state_dict(),parent=a.warm,manual_tail_control=dict(smooth=smooth,outline=outline,tone=tone,majority=a.majority,sharpen=a.sharpen))
        torch.save(child,sub/'initial.pt')
        records.append(dict(name=label,package_sha256=file_hash(sub/'coefficients-padded.bin'),records=records_sub))
        print('FLAT_FIELD_CONTROL',label,flush=True)
    board=Image.new('RGB',(420*(1+len(variants)),314*len(CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    rows={r['id']:r for r in data['validation']}
    for i,(key,box,scene) in enumerate(CROPS):
        cols=[('target',Image.open(rows[key]['target']))]+[(name,images[(name,key)]) for name,_,_,_ in variants]
        for j,(name,im) in enumerate(cols):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+5),scene+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),code_sha256=file_hash(__file__),records=records,
        cpu_actual_integer=True,added_operators=0,postfilter=False,visual_acceptance=False,full_export_verification_pending=True))


if __name__=='__main__':main()
