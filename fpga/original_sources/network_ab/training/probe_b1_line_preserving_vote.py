"""Test line preservation versus isolated colour predicates in existing B1.

The centre-heavy positive kernel is an engineering hypothesis, not a proof of
minimum-region area. Check constant colours, order, source replay and synthetic
thin structures separately. No additional inference operation is introduced.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits,GROUPS30
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

KERNELS={'weighted3':([[1,2,1],[2,3,2],[1,2,1]],8738),
         'centre9':([[1,1,1],[1,9,1],[1,1,1]],7710)}

@torch.no_grad()
def install(net,name):
    kernel,mult=KERNELS[name];key='res3_conv2__dw'
    net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(mult)
    net.weight[key][:30,0]=net.weight[key].new_tensor(kernel)*mult/131072
    key='res3_conv2__pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    for i in range(30):net.weight[key][i,i,0,0]=3*131071/131072
    net.bias[key][:30]=-127*131071/131072

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False);setup=state['binary_coverage720']
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127)
    yy,xx=torch.meshgrid(torch.arange(40),torch.arange(48),indexing='ij')
    patterns={'dot':(xx==24)&(yy==20),'horizontal':yy==20,'vertical':xx==24,
              'diagonal':xx-yy==4,'solid':xx>=24}
    records=[];rows=m.load_data(a.data)['validation'];boards=[]
    for name in KERNELS:
        net=copy.deepcopy(base);install(net,name)
        z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
        assert torch.equal(fragment(net,z,'res3_conv1.pw')['rgb_head'][:,:,0,0],palette)
        for pattern,mask in patterns.items():
            for contrast,(a0,b0) in enumerate(((5,9),(5,6))):
                code=torch.where(mask,b0*81+40,a0*81+40)[None]
                z=F.pad(code_bits(code)*127,(0,0,0,0,0,2));nodes=fragment(net,z,'res3_conv1.pw')
                for lo,hi in GROUPS30:assert (torch.diff(nodes['up1_low_dw'][:,lo:hi],dim=1)<=0).all()
                rgb=nodes['rgb_head'][0];mask720=mask.repeat_interleave(4,0).repeat_interleave(4,1)
                delta=(rgb-palette[a0*81+40][:,None,None]).abs().mean(0)
                interior=(xx>4)&(xx<43)&(yy>4)&(yy<35);valid=interior.repeat_interleave(4,0).repeat_interleave(4,1)
                records.append(dict(variant=name,pattern=pattern,contrast=contrast,
                  centre_delta=float(delta[mask720&valid].mean()),max_delta=float(delta[valid].max()),
                  outside_delta=float(delta[~mask720&valid].mean())))
                m.t.r.c.rgb_image(rgb/128).resize((384,320),Image.Resampling.NEAREST).save(dest/f'{name}_{pattern}_{contrast}.png')
        raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');assert len(raw)==20480;ims={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(dest/f'{name}_{row["id"]}.png');ims[row['id']]=im
        boards.append(ims);saved=copy.deepcopy(state);saved.update(model=net.state_dict(),ordinal_geometry_variant=name,visual_acceptance=False)
        torch.save(saved,dest/(name+'.pt'));print('LINE_PRESERVING_VOTE_PASS',name,'1215 constants/order/6x720 replay/20KiB',flush=True)
    board=Image.new('RGB',(840,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for j,(name,ims) in enumerate(zip(KERNELS,boards)):
        for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
            board.paste(ims[rid].crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'source_crops.png');write_json(dest/'review.json',dict(records=records,constant1215_error=0,
      source_replay_error=0,bytes=20480,extra_operators=0,visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))

if __name__=='__main__':main()
