"""Known-field counterexample for the NEW designed luminance decoder only.

Do not attribute this experiment to the old learned model. Inject two RGB
plateaus at180p, represent them as14Y+CoCg, and measure whether independently
sharpening Y bends the RGB interpolation away from its endpoint segment.
"""
import argparse,copy,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .design_b1_edge_stopping_diffusion import install
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def encode(rgb,net):
    r,g,b=rgb[:,0:1],rgb[:,1:2],rgb[:,2:3]
    y=(r+2*g+b)/4;co=(r-b)/2+64;cg=(-r+2*g-b)/4+64
    thresholds=torch.arange(14)[None,:,None,None]*8+12
    return net.round(torch.cat(((32*(y-thresholds)+63.5).clamp(0,127),co,cg),1)).clamp(0,127)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);_,base=load(a.checkpoint)
    gen=torch.Generator().manual_seed(846);pairs=torch.randint(24,112,(128,2,3),generator=gen).float()
    pairs[0]=torch.tensor([[40,48,80],[102,94,64.]])
    rgb=pairs[:,0,:,None,None].expand(-1,-1,20,64).clone();rgb[...,32:]=pairs[:,1,:,None,None]
    out=Path(a.out);out.mkdir(exist_ok=False);records=[];images=[]
    for gain in (1,2):
        net=copy.deepcopy(base);install(net,decode_gain=gain)
        z=encode(rgb,net);p=fragment(net,z,'up1_low_pw')['rgb_head']+128
        left=p[:,:,40,40];right=p[:,:,40,-40];v=right-left
        band=p[:,:,40,115:141].permute(0,2,1);delta=band-left[:,None]
        alpha=(delta*v[:,None]).sum(-1)/(v.square().sum(-1)[:,None].clamp_min(1))
        ideal=left[:,None]+alpha.clamp(0,1)[...,None]*v[:,None]
        distance=(band-ideal).square().mean(-1).sqrt()
        records.append(dict(gain=gain,rgb_off_segment_mean=float(distance.mean()),
            rgb_off_segment_max=float(distance.max()),pairs=128))
        im=m.t.r.c.rgb_image((p[0]-128)/128);images.append(im)
    assert records[1]['rgb_off_segment_mean']>records[0]['rgb_off_segment_mean']+.1
    board=Image.new('RGB',(512,110),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for j,(im,title) in enumerate(zip(images,('共同线性空间响应·注入诊断','单独加硬明暗·注入诊断'))):
        board.paste(im,(j*256,30));draw.text((j*256+3,4),title,font=font,fill='white')
    board.save(out/'counterexample.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),
        decoder_source_sha256=file_hash(Path(__file__).with_name('design_b1_edge_stopping_diffusion.py')),
        scope='new designed decoder only; target-known feature injection, not source inference',
        old_learned_model_cause_proven=False,visual_acceptance=False))
    print('DESIGNED_COLOUR_RAMP_COUNTEREXAMPLE',records,flush=True)


if __name__=='__main__':main()
