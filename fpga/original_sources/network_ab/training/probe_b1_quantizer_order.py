"""Move colour quantization before the final coverage interpolation in B1.

Same two existing convolutions. Constant region palette is retained to integer
precision; edges get more coverage bits instead of re-rounding coarse colours.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def reorder(net,factor=4):
    weight=net.weight['up2_high_dw'];gains=weight[:,0].sum((1,2)).clone()
    assert (gains>0).all() and float(gains.max())<.3
    net.weight['up2_low_pw']*=gains[:,None,None,None];net.bias['up2_low_pw']*=gains
    weight*=factor/gains[:,None,None,None]
    net.bias['up2_high_dw']*=factor/gains
    net.weight['rgb_head']/=factor
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):
        w=net.weight[key];s=w.abs().flatten(1).amax(1).clamp_min(1e-5)/120
        getattr(net,key+'_m').copy_((s*131072).ceil().clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];d=m.load_data()
    images={r['id']:[Image.fromarray(m.t.r.read(r['target']))] for r in d['validation']}
    for name in ('current','before-aa'):
        if name=='before-aa':reorder(net)
        out=dest/name;out.mkdir();export(net,d['validation'],out/'package')
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image((out/'package/coefficients-padded.bin').read_bytes()))
        for row in d['validation']:
            q=actual(m.t.r.tensor(m.t.r.read(row['path']))[None]*128)/128
            im=m.t.r.c.rgb_image(q[0]);im.save(out/(row['id']+'.png'));images[row['id']].append(im)
        candidate={k:v for k,v in state.items() if k not in ('optimizer','raw_scales','rng','cuda_rng','torch_rng')}
        candidate.update(model=copy.deepcopy(net.state_dict()),quantizer_before_aa=name=='before-aa',parent=a.warm)
        torch.save(candidate,out/'initial.pt')
        print('QUANTIZER_ORDER',name,flush=True)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('目标','原整数包','先定色再边界覆盖·整数包'))):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,real_byte_inference=True,added_operators=0,postfilter=False,visual_acceptance=False))


if __name__=='__main__':main()
