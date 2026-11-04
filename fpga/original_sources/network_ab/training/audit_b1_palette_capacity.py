"""CPU colour-only oracle: can the fixed palette represent target colours?

This bypasses inference and uses target pixels. It is diagnostic, never a
candidate network image. It separates missing colours from wrong region maps.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .minimum_area_blocks import OUT
from .runtime import write_json, file_hash


def round_away(x):return np.sign(x)*np.floor(np.abs(x)+.5)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model']
    scale=state['rgb_head_m'].numpy().reshape(3)/131072
    w=round_away(state['weight.rgb_head'].numpy()[:,:,0,0]/scale[:,None]).clip(-127,127)
    b=round_away(state['bias.rgb_head'].numpy()/scale)
    indices=np.array(list(itertools.product(range(9),range(5),range(5))))
    features=np.stack([np.clip(indices[:,c]-2*g,0,2)*2 for c,groups in enumerate((4,2,2)) for g in range(groups)],1)
    palette=round_away((features@w.T+b)*scale).clip(-128,127)+128
    records=[];images={}
    for row in json.loads((OUT/'data.json').read_text(encoding='utf-8'))['validation']:
        rgb=np.array(Image.open(row['target']).convert('RGB'))
        colours,inverse=np.unique(rgb.reshape(-1,3),axis=0,return_inverse=True)
        distance=((colours[:,None].astype(np.float32)-palette[None].astype(np.float32))**2).sum(2)
        ids=distance.argmin(1);output=palette[ids[inverse]].astype(np.uint8).reshape(rgb.shape)
        delta=np.abs(output.astype(np.float32)-rgb)
        records.append(dict(id=row['id'],mae=float(delta.mean()),p95=float(np.percentile(delta,95)),
            used_palette_states=int(len(np.unique(ids[inverse])))))
        images[row['id']]=[Image.fromarray(rgb),Image.fromarray(output)]
        Image.fromarray(output).save(dest/(row['id']+'_TARGET_ORACLE.png'))
        print('PALETTE_COLOUR_ORACLE',records[-1],flush=True)
    from .train_b1_structural_gate import CROPS
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);board=Image.new('RGB',(840,314*len(CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(CROPS):
        for j,title in enumerate(('离线目标','仅颜色容量诊断：标签直接选色')):
            board.paste(images[key][j].crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(checkpoint=a.checkpoint,checkpoint_sha256=file_hash(a.checkpoint),records=records,
        palette=palette.tolist(),source_inference=False,label_injection=True,visual_acceptance=False))


if __name__=='__main__':main()
