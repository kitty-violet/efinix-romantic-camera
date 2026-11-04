"""Test whether the old4px bright-side allowance hides visible shoulders.

Use differentiable per-region mean centering so whole-region darkening does
not satisfy the diagnostic. A2px allowance exceeds the measured ideal1.65px
half-width. No prediction is filtered, recoloured or changed by this audit.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .train_b1_packed720_local_rims import centred_region_luma
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def rim_terms(p,t,labels,radius=2):
    centred,target=centred_region_luma(p,t,labels)
    high=F.max_pool2d(target,2*radius+1,1,radius)
    excess=F.relu(centred-high-4/128)[...,4:-4,4:-4]
    top=excess.flatten().topk(max(1,int(excess.numel()*.05))).values
    loss=(top.square().mean()+1e-12).sqrt()-1e-6
    return loss,excess


def regression():
    lab=torch.zeros(1,64,96,dtype=torch.long);lab[...,48:]=1
    t=torch.zeros(1,3,64,96);t[...,48:]=.8
    for radius in (2,4):
        assert rim_terms(t,t,lab,radius)[0].abs()<1e-7
        offset=t+torch.where(lab[:,None]==0,.08,-.05)
        assert rim_terms(offset,t,lab,radius)[0].abs()<1e-7
    p=t.clone();p[...,44:48]=.25;p.requires_grad_()
    old= rim_terms(p,t,lab,4)[0];new=rim_terms(p,t,lab,2)[0]
    assert old<1e-7 and new>.05
    new.backward();assert torch.isfinite(p.grad).all() and p.grad[...,44:46].sum()>0
    for region in (0,1):assert (p.grad*(lab==region)[:,None]).sum((2,3)).abs().max()<1e-6
    # Two-pixel geometric displacement is still admitted; sharp moved edges
    # must not be misclassified as a shoulder merely for1px location error.
    p=t.clone();p[...,46:48]=.8
    assert rim_terms(p,t,lab,2)[0].abs()<1e-7
    print('TIGHT_RIM_PASS target0/region_offset0/old_missed_shoulder_detected/gradient_region_sum0/two_pixel_shift',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);regression();out=Path(a.out);out.mkdir(exist_ok=False);records=[];shots={}
    tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
    for row in m.load_data(a.data)['validation']:
        path=Path(a.preview)/(row['id']+'_output.png');rgb=m.t.r.read(path);p=tensor(rgb);t=tensor(m.t.r.read(row['target']))
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        old,eo=rim_terms(p,t,labels,4);new,en=rim_terms(p,t,labels,2)
        newly=(en>4/128)&(eo<=4/128);heat=np.zeros_like(rgb);heat[4:-4,4:-4,0]=(en[0,0].clamp(0,.25).numpy()*1020).astype(np.uint8)
        heat[4:-4,4:-4,1]=newly[0,0].numpy().astype(np.uint8)*200
        rec=dict(id=row['id'],radius4=float(old),radius2=float(new),newly_visible_fraction=float(newly.float().mean()),prediction_sha256=file_hash(path))
        records.append(rec);shots[row['id']]=[Image.fromarray(rgb),Image.fromarray(heat)]
    board=Image.new('RGB',(840,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,name) in enumerate(zip(shots[rid],('实际预测','白肩误差；黄为旧4px容差遗漏'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+2,314*i+4),title+' | '+name,font=font,fill='white')
    board.save(out/'diagnostic.png');write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),prediction_changed=False,visual_acceptance=False))
    print('RIM_TOLERANCE_AUDIT',records,flush=True)


if __name__=='__main__':main()
