"""Target-region injection isolates predictor from fixed renderer limitations.

Oracle panels read the answer and are diagnostic only, never network results.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .seed_b1_deep_regions import THRESHOLDS
from . import train_b1_minimum_area as m
from .probe_b1_cartoon_int8 import half
from .audit_b1_region_shapes import compare
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def tail(actual,z,start):
    values={start:z};begun=False
    for row in actual.plan['nodes']:
        name=row['id']
        if name==start:begun=True;continue
        if not begun:continue
        a=values[row['inputs'][0]]
        if name in actual.ops:
            w,b,mul=actual.ops[name]
            if row['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
            dot=F.conv2d(a,w,stride=row['stride'],groups=row['groups'])
            y=half((dot.double()+b).clamp(-(2**31),2**31-1)*mul/131072).clamp(-128,127).float()
            if row['activation']=='relu':y=y.clamp_min(0)
        elif row['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
        else:
            other=values[row['inputs'][1]];q=actual.package['residual_encoding'][name]
            small,large=(a,other) if q['main_scaled'] else (other,a)
            y=half((half(small.double()*q['multiplier']/131072).clamp(-128,127)+large.double())*q['post_multiplier']/131072).clamp(-128,127).float()
        values[name]=y
    return values['rgb_head']


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100693);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()))
    d=m.load_data();pictures={};records=[]
    for row in d['validation']:
        src=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.read(row['target'])
        nodes,_=actual(src*128,True);replay=tail(actual,nodes['res2_add'],'res2_add');assert torch.equal(replay,nodes['rgb_head'])
        z=m.t.r.tensor(target)[None]*64+64;r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
        colour=torch.cat(((r+2*g+b)/4,(r-b)/2+64,(-r+2*g-b)/4+64),1)
        masks=torch.cat([(colour[:,c:c+1]>v).float() for c,v in THRESHOLDS],1)
        coverage=half(F.avg_pool2d(masks,4)*127);coverage=torch.cat((coverage,torch.zeros_like(coverage)),1)
        outputs=[nodes['rgb_head'],tail(actual,coverage,'res2_add'),tail(actual,coverage,'res3_add')]
        pictures[row['id']]=[Image.fromarray(target)]+[m.t.r.c.rgb_image(p[0]/128) for p in outputs]
        record=dict(id=row['id'])
        for key,p in zip(('actual','oracle_before_cleanup','oracle_after_cleanup'),outputs):
            rgb=np.array(m.t.r.c.rgb_image(p[0]/128));record[key]=dict(mae=float(np.abs(rgb.astype(float)-target).mean()),edges=compare(rgb,target))
        records.append(record)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    titles=['目标','真实网络输出','喂入正确区域：含清理','喂入正确区域：只解码']
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,im in enumerate(pictures[key]):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+titles[j],font=font,fill='white')
    board.save(dest/'diagnostic.png')
    write_json(dest/'review.json',dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,
        diagnostic_only=True,oracle_panels_are_not_deployable=True,actual_tail_replay_error=0))
    print('RENDERER_DIAGNOSTIC',[(r['id'],[round(r[k]['mae'],2) for k in ('actual','oracle_before_cleanup','oracle_after_cleanup')]) for r in records],flush=True)


if __name__=='__main__':main()
