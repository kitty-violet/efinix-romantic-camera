"""Check how preclip margins alter the RGB path of a shared boundary.

Known target injection only. Compare clipped ordinal margins with exact
0/63.5/127 coverage states, using the existing integer decoder unchanged.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ternary_field_targets import canonical_states
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_boundary_profile import boundary_profile
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state['aggregate720'];levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255)
    pk=tuple(map(tuple,palette.tolist()));rows={r['id']:r for r in m.load_data(a.data)['validation']}
    key='up1_high_dw';s=(getattr(net,key+'_m')/131072).float().flatten()
    gain=(net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]).sum((1,2,3))[:15][None,:,None,None]
    bias=(net.round(net.bias[key]/s)*s)[:15][None,:,None,None]
    board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);records=[]
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        row=rows[rid];x0,y0,x1,y1=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        t=torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        lab=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        fields=canonical_states(code);pictures=[('目标',t)];records.append(dict(id=rid,crop=label,variants={}))
        for margin in (128.,80.,63.5):
            wanted=63.5+margin*(fields-1)
            latent=F.pad(net.round(F.avg_pool2d((wanted-bias)/gain,4)).clamp(0,127),(0,0,0,0,0,1))
            p=fragment(net,latent,'up1_low_pw')['rgb_head']/128
            _,v=joint_region_loss(p[...,16:-16,16:-16],t[...,16:-16,16:-16],lab[...,16:-16,16:-16])
            _,v2=boundary_profile(p,t,lab);records[-1]['variants'][str(margin)]={k:float(q) for k,q in (v|v2).items()}
            pictures.append((f'已知目标注入·幅度{margin}',p))
        for j,(name,p) in enumerate(pictures):
            board.paste(m.t.r.c.rgb_image(p[0]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
        print('FIELD_TRANSITION_MARGIN',label,[(k,v['region_flatness'],v['rgb_profile_shape']) for k,v in records[-1]['variants'].items()],flush=True)
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(records=records,source_prediction=False,known_target_injection=True,
        visual_acceptance=False,checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
