"""Localize bright false rims and spatial-grid locking in actual B1 output.

Brightness excess is compared with a neighbourhood of the stylized target and
separately with source neighbourhoods. It is not a semantic white-object label.
The continuous encoder is an explicitly nondeployable causality diagnostic.
"""
import argparse,copy
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_regions import field_pre
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


def brightness(rgb):return rgb[:,0:1]*.25+rgb[:,1:2]*.5+rgb[:,2:3]*.25


def measures(p,t,source):
    y,yt,ys=map(brightness,(p,t,source));max_t=F.max_pool2d(yt,9,1,4);max_s=F.max_pool2d(ys,9,1,4)
    excess=(y-max_t).clamp_min(0);mask=(excess>12)&(y>128)
    new=(y>max_s+12)&mask
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask[0,0].numpy().astype(np.uint8),connectivity=8)
    small=stats[1:,cv2.CC_STAT_AREA]<64
    phase={}
    for axis in (-1,-2):
        edge=torch.diff(p,dim=axis).abs().amax(1)[0];size=edge.shape[axis];pos=torch.arange(size)%4
        mass=[float(edge.narrow(axis,0,size).index_select(axis,torch.where(pos==k)[0]).sum()) for k in range(4)]
        total=max(sum(mass),1e-6);phase[str(axis)]=[v/total for v in mass]
    return dict(bright_target_excess_fraction=float(mask.float().mean()),new_above_source_fraction=float(new.float().mean()),
        bright_small_components=int(small.sum()),bright_small_pixels=int(stats[1:,cv2.CC_STAT_AREA][small].sum()),
        bright_excess_mean=float(excess[mask].mean()) if mask.any() else 0.,grid4_edge_mass=phase),mask


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
    dest=Path(a.out);dest.mkdir(exist_ok=False);state,net=load(a.checkpoint);floating=copy.deepcopy(net);floating.quantized=False
    for k,w in floating.weight.items():
        s=(getattr(floating,k+'_m')/131072).float().flatten();w.copy_(net.round(w/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]);floating.bias[k].copy_(net.round(floating.bias[k]/s)*s)
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');rows=m.load_data(a.data)['validation'];records=[];images={}
    for row in rows:
        def read(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()
        src,t=read(row['path']),read(row['target']);x=src/128-1;nodes=net(x,all_nodes=True);real,_=actual(src-128,True);assert all(torch.equal(v,real[k]) for k,v in nodes.items())
        c=floating(x,all_nodes=True);hybrid=fragment(net,net.round(c['up1_low_pw']),'up1_low_pw')['rgb_head']+128
        pre=field_pre(net,nodes);unrounded=fragment(net,F.pad(pre,(0,0,0,0,0,1)),'up2_low_dw')['rgb_head']+128
        # The last spatial kernel must be nonnegative to be a convex coverage
        # blend; record rather than assume whether it can create an overshoot.
        variants={'actual':nodes['rgb_head']+128,'encoder-rounding-off diagnostic':hybrid,'trit-rounding-off diagnostic':unrounded,'target':t,'source':src}
        record=dict(id=row['id'],variants={})
        for name,p in variants.items():
            stats,mask=measures(p,t,src);record['variants'][name]=stats;images[row['id'],name]=m.t.r.c.rgb_image((p[0]-128)/128)
            if name=='actual':
                rgb=p[0].permute(1,2,0).numpy().clip(0,255).astype(np.uint8);rgb[mask[0,0].numpy()]=[255,0,180];images[row['id'],'bright-excess mask']=Image.fromarray(rgb)
        records.append(record);print('WHITE_RIM_GRID',row['id'],{k:(round(v['bright_target_excess_fraction'],5),v['bright_small_components']) for k,v in record['variants'].items() if k!='source'},flush=True)
    for index,(key,box,label) in enumerate(m.t.r.c.CROPS):
        names=['source','target','actual','bright-excess mask','encoder-rounding-off diagnostic','trit-rounding-off diagnostic'];board=Image.new('RGB',(1260,628),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
        for j,name in enumerate(names):
            xx=j%3*420;yy=j//3*314;board.paste(images[key,name].crop(box),(xx,yy+34));draw.text((xx+3,yy+4),label+' | '+name,font=font,fill='white')
        board.save(dest/f'crop_{index}.png')
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),records=records,actual_all_node_error=0,
        decoder_high_dw_min=float(net.weight['up2_high_dw'].min()),diagnostics_not_deployable=True,visual_acceptance=False,
        limitations='4px target neighbourhood permits small alignment differences but cannot semantically label a highlight. Phase histogram alone does not measure perceived jaggedness.'))


if __name__=='__main__':main()
