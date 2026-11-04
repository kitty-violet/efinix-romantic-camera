"""DC-exact spatial-kernel counterfactuals for synthetic white-rim localization.

Move part of each off-centre integer coefficient to its centre, with row-wise
range limits. No colour/DC calibration, multiplier or graph change. A smaller
halo is not sufficient acceptance; natural-image crops remain visible.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_step_ringing import test_inputs,metrics
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.no_grad()
def move_to_centre(net,key,fraction=.5):
    w=net.weight[key];s=(getattr(net,key+'_m')/131072).float().flatten()
    wi=net.round(w/s[:,None,None,None]).clamp(-127,127)
    off=wi.clone();off[:,:,1,1]=0
    # Each input/output connection retains exactly the same integer DC sum.
    proposed=(off*fraction).round();change=proposed.sum((-2,-1))
    centre=wi[:,:,1,1];allowed=torch.where(change>=0,127-centre,127+centre)
    factor=(allowed/(change.abs()+1e-9)).clamp(0,1)*fraction
    for _ in range(12):
        removed=(off*factor[:,:,None,None]).round();delta=removed.sum((-2,-1))
        bad=(centre+delta).abs()>127
        if not bad.any():break
        factor[bad]*=.8
    assert not bad.any()
    new=wi-removed;new[:,:,1,1]+=delta
    assert new.abs().max()<=127 and torch.equal(wi.sum((-2,-1)),new.sum((-2,-1)))
    w.copy_(new*s[:,None,None,None])
    return dict(integer_weights_changed=int((wi!=new).sum()),dc_error=0,
                offcentre_l1_before=float(off.abs().sum()),offcentre_l1_after=float((off-removed).abs().sum()))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);dest=Path(args.out);dest.mkdir(exist_ok=False)
    state,base=load(args.checkpoint);keys=[]
    for k,w in base.weight.items():
        if k=='up1_high_dw':break
        if w.shape[-1]==3 and k not in ('down1__dw','down2__dw'):keys.append(k)
    src,source_records=test_inputs();src=src[::3]
    ends=torch.stack([torch.tensor(r[k]) for r in source_records[::3] for k in ('left','right')]).float()[:,:,None,None].expand(-1,-1,48,48)
    endpoint=base(ends/128-1)[0][:,:,24,24]*128+128
    data=m.load_data(args.data);byid={r['id']:r for r in data['validation']};samples=[]
    for rid,box,label in m.t.r.c.CROPS:
        x0,y0,x1,y1=box;a=m.t.r.read(byid[rid]['path'])[y0:y1,x0:x1]
        samples.append((rid,label,torch.from_numpy(a.transpose(2,0,1).copy())[None].float()/128-1))
    records=[]
    for key in ['unchanged']+keys:
        net=copy.deepcopy(base);change={} if key=='unchanged' else move_to_centre(net,key)
        current_end=net(ends/128-1)[0][:,:,24,24]*128+128
        assert torch.equal(current_end,endpoint),key
        p=net(src/128-1)[0]*128+128
        mm=metrics(p[...,16:-16,32:-32],endpoint[::2],endpoint[1::2])
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        na=net(src[:1]/128-1,all_nodes=True);nb,_=actual(src[:1]-128,True)
        assert all(torch.equal(v,nb[k]) for k,v in na.items())
        sub=dest/key;sub.mkdir();images=[]
        for rid,label,x in samples:
            im=m.t.r.c.rgb_image(net(x)[0][0]);images.append(im)
        board=Image.new('RGB',(840,628),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
        for i,(im,(_,label,_)) in enumerate(zip(images,samples)):
            xx=i%2*420;yy=i//2*314;board.paste(im,(xx,yy+34));draw.text((xx+3,yy+4),key+' | '+label,font=font,fill='white')
        board.save(sub/'source_crops.png')
        mean={k:float(np.mean([a[k] for a in mm])) for k in mm[0]}
        records.append(dict(key=key,change=change,metrics=mean,cases=mm))
        print('RINGING_KERNEL_ROLE',key,mean,flush=True)
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(args.checkpoint),records=records,
        byte_node_error=0,constant_endpoint_change=0,training_updates=0,accepted=False,
        scope='One-kernel-at-a-time half-offcentre ablation; changes need not preserve natural-image structure'))


if __name__=='__main__':main()
