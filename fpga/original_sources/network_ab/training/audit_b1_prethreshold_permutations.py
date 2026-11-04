"""Audit same-colour field permutations BEFORE the first upsampling.

The older spatial check starts after up2_low_dw; it cannot certify stages
before that point. Sorting below is strictly diagnostic and is never exported.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import GROUPS
from .probe_b1_decoder_coverage import fragment
from .minimum_area_blocks import components
from .audit_b1_native_colour_regions import nearest_codes
from .runtime import write_json, file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    dest=Path(args.out);dest.mkdir(exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    synthetic=[]
    for group,(start,end) in enumerate(GROUPS):
        p=((torch.arange(2**(end-start))[:,None]>>torch.arange(end-start)[None])&1).float()
        left,right=torch.where(p.sum(1)[:,None]==p.sum(1)[None])
        changed=[];maxerror=0
        for chunk in range(0,len(left),128):
            a,b=left[chunk:chunk+128],right[chunk:chunk+128]
            z=torch.zeros(len(a),16,4,4);z[:,:3]=127;z[:,7:9]=127;z[:,11:13]=127
            z[:,start:end]=p[a,:,None,None]*127
            base=fragment(net,z,'up1_low_pw')['rgb_head']
            z[:,start:end,:,2:]=p[b,:,None,None]*127
            out=fragment(net,z,'up1_low_pw')['rgb_head'];err=(out-base).abs()
            maxerror=max(maxerror,float(err.max()));changed.extend((err.flatten(1).amax(1)>0).tolist())
        synthetic.append(dict(group=group,pairs=len(left),affected_pairs=sum(changed),max_rgb_error=maxerror))
    palette=np.array(state['palette'])*128+128;data=m.load_data();pictures={};records=[]
    for row in data['validation']:
        x=torch.tensor(m.t.r.read(row['path']).transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        nodes=net(x,all_nodes=True);raw=nodes['up1_low_pw']
        replay=fragment(net,raw,'up1_low_pw')['rgb_head'];assert torch.equal(replay,nodes['rgb_head'])
        sorted_fields=raw.clone();differ=[]
        for start,end in GROUPS:
            values=raw[:,start:end];sorted_fields[:,start:end]=values.sort(1,descending=True).values
            mask=values>=63.5;count=mask.sum(1)
            for dim in (-1,-2):
                a=mask.narrow(dim,1,mask.shape[dim]-1);b=mask.narrow(dim,0,mask.shape[dim]-1)
                ca=count.narrow(dim,1,count.shape[dim]-1);cb=count.narrow(dim,0,count.shape[dim]-1)
                differ.append(float(((a!=b).any(1)&(ca==cb)).float().mean()))
        diagnostic=fragment(net,sorted_fields,'up1_low_pw')['rgb_head'];rec=dict(id=row['id'],
            same_count_different_pattern_neighbour_fraction=max(differ),fragment_exact=True,
            sorted_diagnostic_pixel_change=float((diagnostic!=replay).any(1).float().mean()))
        for name,value in [('actual',replay),('diagnostic',diagnostic)]:
            im=m.t.r.c.rgb_image(value[0]/128);pictures[name,row['id']]=im
            code,membership=nearest_codes(np.array(im),palette);sizes=np.bincount(components(code).ravel())
            rec[name]=dict(small_area_fraction=float(sizes[sizes<256].sum()/sizes.sum()),**membership)
        records.append(rec);print('PRETHRESHOLD_PERMUTATION',rec,flush=True)
    board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,title) in enumerate([('actual','实际B1'),('diagnostic','离线排序诊断·不能部署')]):
            board.paste(pictures[name,key].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(checkpoint=args.checkpoint,sha256=file_hash(args.checkpoint),
        synthetic=synthetic,records=records,sorting_diagnostic_only=True,exported=False,visual_acceptance=False))


if __name__=='__main__':main()
