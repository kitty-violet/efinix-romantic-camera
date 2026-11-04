"""Isolate the last interpolation footprint with legal B1 coefficients.

Unlike the earlier quarter-mass probe, use enough Q17 range for a centre
coefficient of one. Quantization/DC calibration errors are explicitly logged.
This is a diagnostic, not a claim that nearest upsampling meets image quality.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .seed_b1_complementary_final_basis import deployed
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .audit_b1_rim_tolerance import rim_terms
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def change(net,amount,mean):
    key='up2_high_dw';w,b=deployed(net,key);want=w*(1-amount)
    want[:,0,1,1]+=amount*w.flatten(1).sum(1)
    mult=torch.full((8,),2048.,dtype=torch.float64)
    scale=(mult/131072).float();code=net.round(want/scale[:,None,None,None]);assert code.abs().max()<=127
    new=code*scale[:,None,None,None];drift=(w-new).flatten(1).sum(1)
    net.weight[key].copy_(new);net.bias[key].copy_(b+drift*mean)
    net.up2_high_dw_m.copy_(mult[None,:,None,None])
    return dict(amount=amount,signed_DC_change=(-drift).tolist(),bias_compensation=(drift*mean).tolist(),quantum=scale.tolist())


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data)
    tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
    values=[];ids=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2];ids.append(row['id'])
        z=base(tensor(m.t.r.read(row['path'])[200:520,480:800]),all_nodes=True)['up2_nn']
        values.append(z.mean((0,2,3)))
    mu=torch.stack(values).mean(0);models={'original':base};configs={};records={};ims={}
    for name,amount in (('half_footprint',.5),('point_footprint',1.)):
        net=copy.deepcopy(base);configs[name]=change(net,amount,mu);models[name]=net
    for row in data['validation']:
        x=tensor(m.t.r.read(row['path']));target=tensor(m.t.r.read(row['target']))
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None];ims[row['id']]=[m.t.r.c.rgb_image(target[0])]
        for name,net in models.items():
            nodes=net(x,all_nodes=True);raw=net.export();actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items()) and len(raw)==20480
            p=actual['rgb_head']/128;profile,_=complete_rgb_profiles(p,target,labels);rim,_=rim_terms(p,target,labels,2)
            records.setdefault(name,[]).append(dict(id=row['id'],mae=float((p-target).abs().mean()*128),profile=float(profile),rim=float(rim),actual_all_nodes_error=0))
            ims[row['id']].append(m.t.r.c.rgb_image(p[0]))
        print('FINAL_FOOTPRINT',row['id'],flush=True)
    board=Image.new('RGB',(1680,len(m.t.r.c.CROPS)*314),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(ims[rid],('目标','原实际B1','末级范围收缩一半·诊断','末级中心采样·诊断'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png');write_json(out/'review.json',dict(configs=configs,records=records,calibration_ids=ids,
        source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),graph_unchanged=True,diagnostic_only=True,visual_acceptance=False))


if __name__=='__main__':main()
