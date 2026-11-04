"""Compare averaging colour coordinates versus averaging region indicators.

The two operations need not commute; this diagnostic prevents a tied colour
code from hiding a geometry limitation behind good constant-colour tests.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_binary_layout import fields_for_layout
from .train_b1_ternary_regions import region_codes
from .train_b1_boundary_profile import boundary_profile
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def hinges(coordinate):
    return torch.cat([((coordinate[:,i:i+1]-torch.arange(n,device=coordinate.device)[None,:,None,None])*127).clamp(0,127)
                      for i,n in enumerate((14,8,8))]+[torch.zeros_like(coordinate[:,:2])],1)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);setup=state['binary_coverage720'];dest=Path(a.out);dest.mkdir(exist_ok=False)
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255)
    pk=tuple(map(tuple,palette.tolist()));rows={r['id']:r for r in m.load_data(a.data)['validation']};records=[]
    board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        row=rows[rid];x0,y0,_,_=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        t=torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        lab=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        coordinate=F.avg_pool2d(torch.stack((code//81,code//9%9,code%9),1).float(),4)
        p=fragment(net,net.round(hinges(coordinate)),'res3_conv1.pw')['rgb_head']/128
        score=F.pad(F.avg_pool2d(fields_for_layout(code,setup)*96+16,4),(0,0,0,0,0,1))
        reference=fragment(net,score,'up1_low_pw')['rgb_head']/128
        rec=dict(crop=label,variants={})
        for name,z in [('known_coordinate',p),('known_fields',reference)]:
            _,v=joint_region_loss(z[...,16:-16,16:-16],t[...,16:-16,16:-16],lab[...,16:-16,16:-16]);_,terms=boundary_profile(z,t,lab)
            rec['variants'][name]={k:float(val) for k,val in (v|terms).items()}
        records.append(rec)
        for j,(name,z) in enumerate([('目标',t),('已知三坐标·注入',p),('已知15字段·注入',reference)]):
            board.paste(m.t.r.c.rgb_image(z[0]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    # Exact uniform latent scan, isolated from source encoder and geometry.
    scans=[]
    for axis,maximum in enumerate((14,8,8)):
        v=torch.linspace(0,maximum,maximum*128+1);c=torch.tensor([7.,4.,4.])[None,:,None,None].repeat(len(v),1,1,1);c[:,axis,0,0]=v
        rgb=fragment(net,net.round(hinges(c)),'res3_conv1.pw')['rgb_head'][:,:,0,0]
        direction=torch.tensor(setup['matrix']).float()[:,axis];position=(rgb*direction).sum(1)/direction.square().sum()
        scans.append(dict(axis=axis,min_coordinate_increment=float(torch.diff(position).min()),negative_steps=int((torch.diff(position)<-1e-5).sum())))
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,uniform_scan=scans,source_prediction=False,
        checkpoint_sha256=file_hash(a.checkpoint),visual_acceptance=False));print('COMMON_COORDINATE_BOUNDARIES_DONE',scans,flush=True)


if __name__=='__main__':main()
