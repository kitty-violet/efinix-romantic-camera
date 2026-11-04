"""Actual-byte review on the separate two-source reconstruction test split.

These frames were historically used for reconstruction; this is explicitly
not a blind evaluation. No model update or target injection occurs here.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);package=Path(args.package)
    data=json.loads(Path(args.data).read_text(encoding='utf-8'));rows=data['reserved_test']
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(package.read_bytes()),device='cpu')
    board=Image.new('RGB',(960,300*len(rows)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);records=[]
    for i,row in enumerate(rows):
        assert file_hash(row['path'])==row['sha256']
        rgb=m.t.r.read(row['path']);assert rgb.shape==(720,1280,3)
        x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()-128
        output=m.t.r.c.rgb_image(actual(x)[0]/128);output.save(dest/(row['id']+'.png'))
        for j,(im,title) in enumerate(((Image.fromarray(rgb),'原图'),(output,'实际B1包'))):
            board.paste(im.resize((480,270),Image.Resampling.LANCZOS),(j*480,i*300+30))
            draw.text((j*480+4,i*300+5),row['id']+' | '+title,font=font,fill='white')
        records.append(dict(id=row['id'],source_sha256=row['sha256'],output_sha256=file_hash(dest/(row['id']+'.png'))))
        print('RESERVED_SCENE_REVIEW',row['id'],flush=True)
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(package_sha256=file_hash(package),
        actual_integer_bytes=True,records=records,training_updates=0,blind_test=False,
        scope='Separate two-source split, historically used for reconstruction; not untouched holdout.'))


if __name__=='__main__':main()
