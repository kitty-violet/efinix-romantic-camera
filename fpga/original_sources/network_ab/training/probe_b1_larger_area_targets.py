"""Inspect stronger offline area targets; never label these as B1 outputs."""
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .minimum_area_blocks import make_target,initial_regions
from .runtime import write_json,file_hash


def main():
    cv2.setNumThreads(2);d=m.load_data();dest=m.OUT/'larger-area-target-probe';dest.mkdir(exist_ok=False)
    images={};records=[]
    for row in d['validation']:
        rgb=m.t.r.read(row['parent_target']);initial=initial_regions(rgb)
        images[row['id']]=[Image.open(row['path']).convert('RGB'),Image.open(row['target']).convert('RGB')]
        for area in (512,1024):
            out,labels,stats=make_target(rgb,area,initial)
            image=Image.fromarray(out);path=dest/f'{row["id"]}_{area}_target.png';image.save(path)
            images[row['id']].append(image)
            records.append(dict(id=row['id'],source=str(row['parent_target']),source_sha256=file_hash(row['parent_target']),
                area=area,stats=stats,output_sha256=file_hash(path)))
            print('LARGER_AREA_TARGET',row['id'],area,stats['final720'],flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('原片','离线目标≥256','离线目标≥512','离线目标≥1024'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,network_output=False,training_data_changed=False,
        visual_acceptance=False,minimum_area_is_offline_only=True))


if __name__=='__main__':main()
