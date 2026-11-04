"""Localize final spatial filtering using exactly DC-preserving integer edits.

Diagnostic only: move a bounded amount of positive off-centre kernel mass to
the centre inside the existing DW. No sharpening filter is added to output.
"""
import argparse,copy
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .audit_b1_rim_tolerance import rim_terms
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def concentrate(net,fraction):
    key='up2_high_dw';s=(net.up2_high_dw_m/131072).float().flatten();w=net.round(net.weight[key]/s[:,None,None,None]).clamp(-127,127)
    original=w.flatten(1).sum(1).clone();moved=[]
    for c in range(8):
        v=w[c,0].flatten();room=int(127-v[4]);count=0
        for j in sorted((i for i in range(9) if i!=4),key=lambda i:-float(v[i])):
            take=min(room,max(0,int(round(float(v[j])*fraction))));v[j]-=take;v[4]+=take;room-=take;count+=take
        moved.append(count)
    assert torch.equal(w.flatten(1).sum(1),original)
    net.weight[key].copy_(w*s[:,None,None,None]);return moved


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False);models={'original':base};moves={}
    for name,f in (('concentrate_125',.125),('concentrate_250',.25)):
        net=copy.deepcopy(base);moves[name]=concentrate(net,f);models[name]=net
    rows=m.load_data(a.data)['validation'];ims={};results={k:[] for k in models}
    for row in rows:
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target']);ll=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
        x,t=tensor(source),tensor(target);ims[row['id']]=[Image.fromarray(target)]
        for name,net in models.items():
            nn=net(x,all_nodes=True);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');actual,_=rt(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nn.items());p=actual['rgb_head']/128
            assert len(net.export())==20480
            rim,_=rim_terms(p,t,ll,2);profile,_=complete_rgb_profiles(p,t,ll)
            results[name].append(dict(id=row['id'],mae=float((p-t).abs().mean()*128),rim=float(rim),profile=float(profile),all_nodes_error=0))
            ims[row['id']].append(m.t.r.c.rgb_image(p[0]))
        print('FINAL_DW_PROBE',row['id'],flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(ims[rid],('目标','实际原系数','末端核集中1/8·诊断','末端核集中1/4·诊断'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png');write_json(out/'review.json',dict(results=results,moved_integer_kernel_codes=moves,kernel_DC_unchanged=True,
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),diagnostic_only=True,visual_acceptance=False))


if __name__=='__main__':main()
