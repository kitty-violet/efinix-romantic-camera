"""TRAIN-only intervention: remove texture while preserving labelled shape.

The input intervention requires offline target labels. It is deliberately not
a deployable result. It tests whether incorrect fields persist even after
within-region source texture has been removed; no parameters are updated.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


def region_mean_source(x,labels):
    result=[]
    for xx,ll in zip(x,labels):
        _,ids,count=torch.unique(ll.flatten(),sorted=True,return_inverse=True,return_counts=True)
        indices=ids[None].expand(3,-1)
        mean=xx.new_zeros(3,len(count)).scatter_add(1,indices,xx.flatten(1))/count
        result.append(mean[:,ids].reshape_as(xx))
    return torch.stack(result)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);raw=net.export()
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    rows=[];seen=set()
    for row in m.load_data(args.data)['training']:
        if row['video'] not in seen:seen.add(row['video']);rows.append(row)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);board=Image.new('RGB',(1520,340*len(rows)),'#17202a')
    draw=ImageDraw.Draw(board);records=[]
    for i,row in enumerate(rows):
        def tensor(path):return torch.from_numpy(m.t.r.read(path)[168:552,448:832].transpose(2,0,1).copy())[None].float()/128-1
        x,t=tensor(row['path']),tensor(row['target'])
        labels=torch.from_numpy(m.load_region(row['region'])[168:552,448:832].astype(np.int64))[None]
        cf=region_mean_source(x,labels);p=actual(x*128)/128;pc=actual((cf*128).round())/128
        values={}
        for name,z in [('source',p),('label_mean_input',pc)]:
            loss,v=joint_region_loss(z[...,40:-40,40:-40],t[...,40:-40,40:-40],labels[...,40:-40,40:-40])
            values[name]=dict(loss=float(loss),**{k:float(a) for k,a in v.items()})
        for j,(title,z) in enumerate([('原输入',x),('去纹理输入·目标标签干预',cf),('原输入B1',p),('去纹理输入B1·非部署',pc),('目标',t)]):
            board.paste(m.t.r.c.rgb_image(z[0,...,40:-40,40:-40]),(j*304,i*340+36))
            draw.text((j*304+3,i*340+6),title,font=font,fill='white')
        records.append(dict(id=row['id'],values=values));print('TEXTURE_INTERVENTION',row['id'],values,flush=True)
    board.save(dest/'train-diagnostic.png');write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(args.checkpoint),
        parameter_updates=0,source_prediction=False,training_only=True,requires_target_labels=True,records=records))


if __name__=='__main__':main()
