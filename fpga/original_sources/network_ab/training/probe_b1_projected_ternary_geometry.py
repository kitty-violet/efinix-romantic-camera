"""Diagnostic: can cleaning predicted connected colour codes fix the image?

Reuses training-only connected-area projection from prior experiments, now
on the exact1215-colour code at360p. Projected results are NOT deployment
outputs. A no-change code reconstruction must equal the real decoder first.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components,area_merge,boundary_mode
from .probe_b1_ternary_field_targets import canonical_states,palette
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


def clean_codes(codes,pal,minimum=64):
    original=components(codes);rgb=pal[codes].astype(np.uint8)
    labels,_=area_merge(original,rgb,minimum)
    labels=boundary_mode(labels,1)
    labels,_=area_merge(components(labels),rgb,minimum)
    count=np.bincount(labels.ravel()).clip(1)
    means=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel())/count for c in range(3)],1)
    nearest=((means[:,None]-pal[None])**2).sum(2).argmin(1)
    return nearest[labels].astype(np.int64),dict(original_regions=int(original.max())+1,
        projected_regions=int(labels.max())+1,minimum_label_area=int(count.min()))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);net=m.t.r.c.ColourB1(revive=False)
    net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    pal=np.asarray(palette(net),dtype=np.float64);data=m.load_data(a.data);records=[];pictures={}
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        nodes=net(x,all_nodes=True);z=nodes['up2_low_dw'][:,:15]
        codes=(81*z[:,:7].sum(1)+9*z[:,7:11].sum(1)+z[:,11:15].sum(1)).long()
        for minimum in (0,32,64):
            if minimum:
                cc,geometry=clean_codes(codes[0].numpy(),pal,minimum);cc=torch.from_numpy(cc)[None]
            else:cc=codes;geometry={}
            fields=F.pad(canonical_states(cc),(0,0,0,0,0,1))
            p=fragment(net,fields,'up2_low_dw')['rgb_head']/128
            if minimum==0:assert torch.equal(p*128,nodes['rgb_head'])
            loss,terms=joint_region_loss(p,target,lab);im=m.t.r.c.rgb_image(p[0]);pictures[minimum,row['id']]=im
            im.save(dest/f'{row["id"]}_min{minimum}.png')
            records.append(dict(id=row['id'],minimum360=minimum,changed_fraction=float((cc!=codes).float().mean()),
                                **geometry,loss=float(loss),**{k:float(v) for k,v in terms.items()}))
        print('PROJECTED_TERNARY_DIAGNOSTIC',row['id'],flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);byid={r['id']:r for r in data['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('target',Image.open(byid[key]['target']))]+[(f'projected min{k}' if k else 'actual source prediction',pictures[k,key]) for k in (0,32,64)]
        for j,(title,img) in enumerate(choices):
            board.paste(img.crop(box),(420*j,i*314+34));draw.text((420*j+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,
        checkpoint_sha256=file_hash(a.checkpoint),no_change_reconstruction_exact=True,
        projected_images_are_not_deployable=True,training_updates=0,visual_acceptance=False))


if __name__=='__main__':main()
