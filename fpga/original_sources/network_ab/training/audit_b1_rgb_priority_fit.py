"""Controlled three-TRAIN-crop integer fit: raw versus RGB-priority gradients.

The same initialization, samples, frozen decoder, optimizer and step budget
are used. This diagnoses local optimization only, not style generalization.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_coincident_distance import coincident_loss
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_reachable_loss_fixedpoint import objective
from .train_b1_rgb_priority_distance import gradient_correction
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=100);ap.add_argument('--distance-only',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);state,initial=load(a.checkpoint);initial.__class__=OrdinalBandB1
    dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data(a.data);videos=sorted({r['video'] for r in data['training']});rows=[]
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    xs=[];labs=[];codes=[]
    for video in (videos[0],videos[len(videos)//2],videos[-1]):
        rr=[r for r in data['training'] if r['video']==video];row=rr[len(rr)//2];rows.append(row);sl=np.s_[232:488,512:768]
        xs.append(torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
        labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        codes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
    x=torch.stack(xs);labels=torch.stack(labs);truth=distance_fields(torch.stack(codes),8)
    with torch.no_grad():target=fragment(initial,F.pad(initial.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
    cut=lambda z:z[...,40:-40,40:-40]
    records=[];images=[target]
    for mode in (('distance_only',) if a.distance_only else ('raw','rgb_priority')):
        net=copy.deepcopy(initial);groups=[]
        for k,w in net.weight.items():
            fixed=k.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) or k in ('down1__dw','down2__dw')
            w.requires_grad_(not fixed);net.bias[k].requires_grad_(not fixed)
            if not fixed:
                scale=(getattr(net,k+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=.00015,row_gain=(scale*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.008)])
        opt=RowScaledAdam(groups);parameters=[p for p in net.parameters() if p.requires_grad];history=[]
        for step in range(a.steps+1):
            nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128;field=nodes['up1_low_dw'][...,10:-10,10:-10];t=truth[...,10:-10,10:-10]
            rgb,v=objective(cut(p),cut(target),cut(labels),reference_boundary_profile)
            full,band=distance_objective(field,t);aux=16*full+8*band+10*rank_loss(field)+32*coincident_loss(field,t)
            if step%25==0 or step==a.steps:
                row=dict(step=step,rgb_loss=float(rgb.detach()),auxiliary_loss=float(aux.detach()),distance_full=float(full.detach()),distance_band=float(band.detach()),mae=float((cut(p)-cut(target)).abs().mean().detach()*128),
                    shape=float(v['rgb_profile_shape'].detach()),contrast=float(v['rgb_profile_contrast'].detach()))
                history.append(row);print('RGB_PRIORITY_FIT',mode,row,flush=True)
            if step==a.steps:break
            loss=rgb+aux
            if mode=='distance_only':loss=16*full+8*band
            if mode=='rgb_priority':loss+=gradient_correction(rgb,aux,parameters)[0]
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for k,w in net.weight.items():
                    if w.requires_grad:
                        lim=127*(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];w.copy_(w.maximum(-lim).minimum(lim))
        with torch.no_grad():
            final=net(x)[0];actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');nn,ss=actual(net.round(x*128),True)
            assert torch.equal(nn['rgb_head'],net(x,all_nodes=True)['rgb_head']) and len(net.export())==20480
        images.append(final.detach());records.append(dict(mode=mode,history=history,integer_rgb_error=0,bytes=20480))
    board=Image.new('RGB',(256*len(images),870),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i in range(3):
        for j,(name,p) in enumerate(zip(('已知目标','只拟合距离字段') if a.distance_only else ('已知目标','原训练梯度','保护RGB梯度'),images)):
            board.paste(m.t.r.c.rgb_image(p[i]),(j*256,i*290+34));draw.text((j*256+2,i*290+4),name,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,train_ids=[r['id'] for r in rows],steps=a.steps,
        scope='Controlled local TRAIN fit, no augmentation or temporal term in either arm; not generalization',checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))


if __name__=='__main__':main()
