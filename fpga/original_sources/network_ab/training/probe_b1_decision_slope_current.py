"""Reuse the earlier slope diagnostic on today's free RGB decoder.

This is not a new sharpening invention: it tests whether present soft fields
still dominate blur after releasing the fixed palette. Only existing Q17
weights/biases change. Larger slopes can worsen phase noise and false bands.
"""
import argparse,copy,shutil
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
def harden(net,gain):
    key='up2_low_dw';w,b=deployed(net,key)
    net.weight[key][:14].copy_(w[:14]*gain)
    net.bias[key][:14].copy_((b[:14]-63.5)*gain+63.5)
    net.up2_low_dw_m[:,:14].mul_(gain)
    assert net.up2_low_dw_m.max()<=131071
    s=(net.up2_low_dw_m/131072).float().flatten()
    assert net.round(net.weight[key]/s[:,None,None,None]).abs().max()<=127


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data)
    tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
    models={'original':base};results={};ims={}
    for gain in (2,4):
        net=copy.deepcopy(base);harden(net,gain);models['gain'+str(gain)]=net
    for row in data['validation']:
        x=tensor(m.t.r.read(row['path']));target=tensor(m.t.r.read(row['target']))
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None];ims[row['id']]=[m.t.r.c.rgb_image(target[0])]
        for name,net in models.items():
            nodes=net(x,all_nodes=True);raw=net.export();actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items()) and len(raw)==20480
            p=actual['rgb_head']/128;profile,_=complete_rgb_profiles(p,target,labels);rim,_=rim_terms(p,target,labels,2)
            values=nodes['up2_low_dw'][:,:14]
            results.setdefault(name,[]).append(dict(id=row['id'],mae=float((p-target).abs().mean()*128),profile=float(profile),rim=float(rim),
                any_partial_field=float(((values>0)&(values<127)).any(1).float().mean()),all_nodes_error=0))
            ims[row['id']].append(m.t.r.c.rgb_image(p[0]))
        print('CURRENT_SLOPE',row['id'],flush=True)
    board=Image.new('RGB',(1680,len(m.t.r.c.CROPS)*314),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(ims[rid],('目标','当前实际B1','现有判定斜率×2·诊断','现有判定斜率×4·诊断'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(records=results,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),
        graph_unchanged=True,diagnostic_only=True,visual_acceptance=False))


if __name__=='__main__':main()
