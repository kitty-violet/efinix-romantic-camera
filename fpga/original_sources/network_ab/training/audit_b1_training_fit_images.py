"""Distinguish failure to fit TRAIN images from a validation-only domain gap."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    d=m.load_data(a.data);groups={}
    for row in d['training']:groups.setdefault(row['video'],[]).append(row)
    rows=[v[len(v)//2] for v in groups.values()];pictures={};records=[]
    for j,path in enumerate(a.checkpoints):
        state=torch.load(path,map_location='cpu',weights_only=True)
        net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model'])
        net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
        net.quantized=bool(state.get('qat',state.get('args',{}).get('qat',False)))
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu') if net.quantized else None
        for row in rows:
            def tensor(p):return torch.from_numpy(m.t.r.read(p).transpose(2,0,1).copy())[None].float()/128-1
            x,t=tensor(row['path']),tensor(row['target']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            p=actual(x*128)/128 if actual is not None else net(x)[0]
            total,terms=joint_region_loss(p,t,labels)
            records.append(dict(checkpoint=path,id=row['id'],qat=net.quantized,mae=float((p-t).abs().mean()*128),
                                **{k:float(v) for k,v in terms.items()},score=float(total)))
            im=m.t.r.c.rgb_image(p[0]);im.save(dest/f'{j}_{row["id"]}.png');pictures[j,row['id']]=im
        print('TRAIN_FIT_CHECKPOINT',j,net.quantized,flush=True)
    width=320*(len(a.checkpoints)+2);board=Image.new('RGB',(width,354*len(rows)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,row in enumerate(rows):
        choices=[('TRAIN source',Image.open(row['path'])),('TRAIN target',Image.open(row['target']))]
        choices.extend((f'candidate {j}',pictures[j,row['id']]) for j in range(len(a.checkpoints)))
        line=Image.new('RGB',(width,354),'#17202a');ld=ImageDraw.Draw(line)
        for j,(title,im) in enumerate(choices):
            line.paste(im.crop((480,200,800,520)),(j*320,34));ld.text((j*320+3,5),row['id'][-22:]+' | '+title,font=font,fill='white')
        line.save(dest/f'train_row_{i}.png');board.paste(line,(0,i*354))
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,training_ids=[r['id'] for r in rows],
        checkpoints=[dict(path=p,sha256=file_hash(p)) for p in a.checkpoints],
        inference='native full720; centre crops only for display',training_updates=0))


if __name__=='__main__':main()
