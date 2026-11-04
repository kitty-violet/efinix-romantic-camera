"""Diagnostic: can B1 decode a signed contour plus two neighbouring colours?

Target-derived latent fields are injected only to test decoder capacity.
These images MUST NOT be described as network predictions. The decoder uses
the existing16->8->RGB path, with hard gating at360p and AA at720p.
"""
import argparse
import shutil
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .runtime import configure_workspace,write_json


@lru_cache(maxsize=12)
def latent_full(target_path,region_path):
    lab=m.load_region(region_path).astype(np.int32);rgb=m.t.r.read(target_path).astype(np.float32)
    count=np.bincount(lab.ravel()).clip(1)
    palette=np.stack([np.bincount(lab.ravel(),weights=rgb[...,c].ravel())/count for c in range(3)],1)
    other=lab.copy();boundary=np.zeros(lab.shape,bool)
    padded=np.pad(lab,1,mode='edge');h,w=lab.shape
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        neighbour=padded[1+dy:1+dy+h,1+dx:1+dx+w]
        take=(neighbour!=lab)&~boundary;other[take]=neighbour[take];boundary|=take
    if not boundary.any():
        field=np.concatenate((rgb/2,rgb/2,np.full((h,w,1),64,np.float32)),2)
        return field.astype(np.float32)
    distance,nearest=cv2.distanceTransformWithLabels((~boundary).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    edge_own=lab[boundary];edge_other=other[boundary]
    idx=nearest-1;neighbour=edge_other[idx]
    valid=edge_own[idx]==lab;neighbour=np.where(valid,neighbour,lab)
    own=palette[lab];adjacent=palette[neighbour]
    # Stable role selection from colour ordering, not arbitrary region IDs.
    key=palette@np.array([.25,.5,.25])+palette[:,0]*1e-5+palette[:,2]*1e-7
    positive=key[lab]>=key[neighbour]
    a=np.where(positive[...,None],adjacent,own);b=np.where(positive[...,None],own,adjacent)
    # Beyond the local edge band, both colours equal the region colour.
    # Predicting a distant invisible neighbour is not a training requirement.
    far=distance>12;a=np.where(far[...,None],own,a);b=np.where(far[...,None],own,b)
    gate=64+np.where(positive,1,-1)*(distance+.5)*4
    gate=np.where(far,64,gate).clip(0,127)
    field=np.concatenate((a/2,b/2,gate[...,None]),2).astype(np.float32)
    return field


@lru_cache(maxsize=48)
def latent(target_path,region_path):
    field=latent_full(target_path,region_path);h,w=field.shape[:2]
    return cv2.resize(field,(w//4,h//4),interpolation=cv2.INTER_AREA).transpose(2,0,1).copy()


@torch.no_grad()
def install(net):
    keys=('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['up1_high_dw'][:7,0]=kernel
    net.weight['up2_low_dw'][:6,0,1,1]=1
    net.weight['up2_low_dw'][6,0,1,1]=127*(131071/131072)
    net.bias['up2_low_dw'][6]=-63*127*(131071/131072)
    for c in range(3):
        net.weight['up2_low_pw'][c,c,0,0]=1;net.weight['up2_low_pw'][c,6,0,0]=-1
        net.weight['up2_low_pw'][c+3,c+3,0,0]=1;net.weight['up2_low_pw'][c+3,6,0,0]=1
        net.bias['up2_low_pw'][c+3]=-127
        for j in (c,c+3):
            net.weight['up2_high_dw'][j,0]=.5*kernel;net.weight['up2_high_dw'][j,0,1,1]+=.5
            net.weight['rgb_head'][c,j,0,0]=2
    net.bias['rgb_head'].fill_(-128)
    for k in keys:
        w=net.weight[k];scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    net.up2_low_dw_m[:,6]=131071


@torch.no_grad()
def install_late(net):
    """A + ReLU(B-A+d) - ReLU(d), evaluated in the existing720p DW.

    Seven channels suffice: three A values, three B-A+d values and shared d.
    An offset preserves signed values across the preceding360p ReLU. Each
    WARNING: this historical formula is NOT range preserving. For example,
    RGB A=[60,180,60], B=[180,60,60], gate64 can produce [184,184,64].
    Kept only to reproduce prior diagnostics; do not use as a safe decoder.
    probe_b1_bounded_pair720 tests a different, bounded scalar formula.
    """
    install(net)
    keys=('up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    net.weight['up2_low_dw'][:7,0,1,1]=1
    net.weight['up2_low_dw'][6,0,1,1]=4;net.bias['up2_low_dw'][6]=-192
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(3):
        net.weight['up2_low_pw'][c,c,0,0]=.25
        net.weight['up2_low_pw'][c+3,c+3,0,0]=.25;net.weight['up2_low_pw'][c+3,c,0,0]=-.25
        net.weight['up2_low_pw'][c+3,6,0,0]=.5;net.bias['up2_low_pw'][c+3]=32
        net.weight['up2_high_dw'][c,0]=kernel;net.weight['up2_high_dw'][c+3,0]=kernel
        net.bias['up2_high_dw'][c+3]=-64
        net.weight['rgb_head'][c,c,0,0]=8;net.weight['rgb_head'][c,c+3,0,0]=8;net.weight['rgb_head'][c,6,0,0]=-8
    net.weight['up2_low_pw'][6,6,0,0]=.5;net.bias['up2_low_pw'][6]=32
    net.weight['up2_high_dw'][6,0]=kernel;net.bias['up2_high_dw'][6]=-64
    net.bias['rgb_head'].fill_(-128)
    for k in keys:
        w=net.weight[k];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();cv2.setNumThreads(2);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    net=m.t.r.initialize();net.quantized=True;install(net)
    # Exhaust all gate codes for randomly chosen two-colour inputs. With hard
    # gate each RGB must come from one side, never both or an empty black hole.
    gen=torch.Generator(device='cuda').manual_seed(10609)
    z=torch.zeros(128,16,1,1,device='cuda');z[:,:6]=torch.randint(0,128,(128,6,1,1),generator=gen,device='cuda')
    z[:,6,0,0]=torch.arange(128,device='cuda')
    q=fragment(net,z,'up1_high_dw');chosen=torch.where((z[:,6:7]>=64),z[:,3:6],z[:,:3])
    wanted=(2*chosen-128).repeat_interleave(2,-2).repeat_interleave(2,-1)
    err=float((q['rgb_head']-wanted).abs().max());assert err==0,err
    rows={r['id']:r for r in m.load_data()['validation']};images={};records=[]
    for row in rows.values():
        code=torch.from_numpy(latent(row['target'],row['region'])).cuda()[None].round().clamp(0,127)
        code=torch.nn.functional.pad(code,(0,0,0,0,0,9))
        q=fragment(net,code,'up1_low_pw')['rgb_head']/128;im=m.t.r.c.rgb_image(q[0]);im.save(dest/(row['id']+'_injected.png'))
        target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        records.append(dict(id=row['id'],mae=float((q-target).abs().mean()*128)))
        images[row['id']]=[Image.fromarray(m.t.r.read(row['target'])),im]
    board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('区域目标','注入已知轮廓与两侧颜色·非模型预测'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+6),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,gate_codes_checked=128,selector_error=err,
        target_latents_injected=True,source_prediction=False,trained=False,added_operators=0,visual_acceptance=False))
    print('TWO_COLOUR_INJECTION',records,flush=True)


if __name__=='__main__':main()
