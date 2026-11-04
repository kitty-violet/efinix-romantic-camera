"""Can an existing learned B1 render the desired clean targets themselves?

Target-fed reconstruction is diagnostic; it is not source-to-style inference.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--fp32',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(20261006101);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=not a.fp32
    rows=m.load_data()['validation'];records=[];images={}
    for row in rows:
        target=m.t.r.tensor(m.t.r.read(row['target']))[None];p=net(target)[0]
        records.append(dict(id=row['id'],**m.t.r.metrics(p,target)))
        im=m.t.r.c.rgb_image(p[0]);im.save(dest/(row['id']+'_target_reconstruction.png'))
        images[row['id']]=[Image.fromarray(m.t.r.read(row['target'])),im]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],['目标输入（诊断）','B1重建目标（不是风格推理）'])):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(records=records,checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),fp32=a.fp32,
        source_to_style_result=False,diagnostic_only=True))
    print('TARGET_RECONSTRUCTION',np.mean([r['psnr'] for r in records]),np.mean([r['mae'] for r in records]),flush=True)


if __name__=='__main__':main()
