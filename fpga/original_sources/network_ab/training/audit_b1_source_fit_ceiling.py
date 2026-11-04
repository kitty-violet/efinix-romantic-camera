"""Deliberately overfit three TRAIN source crops to test encoder learnability.

This is not generalization or accepted style. The real source, same B1 weights,
and fixed sharp decoder are used. No labels enter inference. CPU-only so the
independent full-data GPU experiment can continue without resource overlap.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .train_b1_ternary_regions import region_codes
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordinal_transition_band import transition_band_loss
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_quantum_regions import RowScaledAdam
from .audit_b1_optimized_decoder_ceiling import decode as exact_decode
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=500);ap.add_argument('--float-encoder',action='store_true');ap.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026100709);state,net=load(a.checkpoint);net.__class__=OrdinalBandB1;dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state['binary_coverage720'];levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255);pk=tuple(map(tuple,palette.tolist()))
    d=m.load_data(a.data);videos=sorted({r['video'] for r in d['training']});rows=[]
    for video in (videos[0],videos[len(videos)//2],videos[-1]):
        options=[r for r in d['training'] if r['video']==video];rows.append(options[len(options)//2])
    xs=[];truth=[];original=[]
    for row in rows:
        sl=np.s_[232:488,512:768]
        xs.append(torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
        original.append(torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy()).float()/128-1)
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        truth.append(F.avg_pool2d(code_bits(code),4)[0])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=net.to(a.device);x=torch.stack(xs).to(a.device);truth=torch.stack(truth).to(a.device);z=F.pad(net.round(truth*127),(0,0,0,0,0,2))
    with torch.no_grad():target=fragment(net,z,'up1_low_dw')['rgb_head']/128;initial=net(x)[0]
    groups=[];fixed={}
    for key,w in net.weight.items():
        freeze=key.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) or key in ('down1__dw','down2__dw')
        b=net.bias[key];w.requires_grad_(not freeze);b.requires_grad_(not freeze)
        if freeze:fixed[key]=(w.detach().clone(),b.detach().clone())
        else:
            scale=(getattr(net,key+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=.00015,row_gain=(scale*256)[:,None,None,None]),dict(params=[b],lr=.008)])
    opt=RowScaledAdam(groups);history=[]
    net.quantized=not a.float_encoder
    for step in range(a.steps):
        nodes=net(x,all_nodes=True);field=nodes['up1_low_dw']
        p=exact_decode(net,field) if a.float_encoder else nodes['rgb_head']/128
        area=F.smooth_l1_loss(field[:,:30,...,5:-5,5:-5]/127,truth[...,5:-5,5:-5],beta=.08)
        band=transition_band_loss(field[...,5:-5,5:-5],truth[...,5:-5,5:-5]);rank=rank_loss(field)
        pixel=F.smooth_l1_loss(p[...,20:-20,20:-20],target[...,20:-20,20:-20],beta=.04)
        loss=8*area+4*band+2*pixel+10*rank;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),20);opt.step()
        with torch.no_grad():
            for key,w in net.weight.items():
                if w.requires_grad:
                    limit=127*(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None];w.copy_(w.maximum(-limit).minimum(limit))
        if (step+1)%100==0:
            history.append(dict(step=step+1,loss=float(loss.detach()),area=float(area.detach()),band=float(band.detach()),rgb_mae=float((p.detach()-target).abs().mean()*128)))
            print('THREE_TRAIN_CROP_OVERFIT',history[-1],flush=True)
    with torch.no_grad():
        nodes=net(x,all_nodes=True);final=exact_decode(net,nodes['up1_low_dw']) if a.float_encoder else nodes['rgb_head']/128
        for key,(w,b) in fixed.items():assert torch.equal(w,net.weight[key]) and torch.equal(b,net.bias[key])
        net.quantized=True;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device=a.device);other,_=actual(x*128,True);nodes=net(x,all_nodes=True)
        assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        final_actual=nodes['rgb_head']/128
    board=Image.new('RGB',(1280,3*290),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,row in enumerate(rows):
        for j,(title,p) in enumerate((('TRAIN源图',x),('固定解码目标',target),('初始源预测',initial),('FP32编码诊断' if a.float_encoder else '3块QAT·非泛化',final),('同权重整数重载',final_actual))):
            im=m.t.r.c.rgb_image(p[i]);im.save(dest/f'{i}_{j}.png');board.paste(im,(256*j,290*i+34));draw.text((256*j+2,290*i+5),title,font=font,fill='white')
    board.save(dest/'diagnostic.png');saved=copy.deepcopy(state);saved.update(model=net.state_dict(),overfit_diagnostic_only=True,float_encoder_training=a.float_encoder,visual_acceptance=False);torch.save(saved,dest/'diagnostic_only.pt')
    write_json(dest/'review.json',dict(history=history,train_ids=[r['id'] for r in rows],box=[512,232,768,488],
      initial_mae=float((initial-target).abs().mean()*128),final_mae=float((final-target).abs().mean()*128),
      final_actual_mae=float((final_actual-target).abs().mean()*128),float_encoder_training=a.float_encoder,device=a.device,tf32=False,
      actual_all_node_error=0,source_sha256=file_hash(a.checkpoint),generalization_test=False,visual_acceptance=False))

if __name__=='__main__':main()
