"""Check canonical three-state targets through the frozen real decoder.

Known target labels are injected at up1_low_pw solely to test representation;
these pictures are not source predictions and must never be delivered as such.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_ternary_regions import region_codes
from .build_b1_ordered_palette import GROUPS
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


def canonical_states(codes):
    levels=(codes//81,codes//9%9,codes%9)
    return torch.cat([(v[:,None]-2*torch.arange(b-a,device=codes.device)[None,:,None,None]).clamp(0,2)
        for v,(a,b) in zip(levels,GROUPS)],1).float()


@torch.inference_mode()
def palette(net):
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9))
    codes=levels[:,0]*81+levels[:,1]*9+levels[:,2]
    z=F.pad(canonical_states(codes[:,None,None]),(0,0,0,0,0,1))
    p=fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]+128
    return tuple(map(tuple,p.tolist()))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);pk=palette(net)
    rows=m.load_data(args.data)['validation'];pictures={};records=[]
    for row in rows:
        t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        codes=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        fields=canonical_states(codes)
        tests=[]
        for margin in (16,32):
            pre=F.avg_pool2d((fields-1)*margin+63.5,4)
            pre=F.pad(net.round(pre).clamp(0,127),(0,0,0,0,0,1))
            p=fragment(net,pre,'up1_low_pw')['rgb_head']/128
            loss,terms=joint_region_loss(p,t,labels);im=m.t.r.c.rgb_image(p[0]);pictures[row['id'],margin]=im
            im.save(dest/f'{row["id"]}_injected{margin}.png')
            tests.append(dict(margin=margin,loss=float(loss),**{k:float(v) for k,v in terms.items()}))
        records.append(dict(id=row['id'],tests=tests))
        print('TERNARY_FIELD_TARGET',row['id'],tests,flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(title,im) in enumerate([('target',Image.open(by_id[key]['target'])),('injected16',pictures[key,16]),('injected32',pictures[key,32])]):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(args.checkpoint),
        injected_target_labels=True,source_prediction=False,parameter_updates=0,records=records))


if __name__=='__main__':main()
