"""Nearest-neighbour crops to make four-pixel cells visible in review."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json
from library import D,OLD
from render_review import FONT

def main():
    root=D/'experiments/square-flat-v1';font=ImageFont.truetype(FONT,21)
    src=Image.open(OLD/'acceptance-four-scenes-5s-20261029/originals/outdoor_076.png').convert('RGB')
    box=(800,120,1120,360);size=(640,480)
    rows=[]
    for n in [4,8,16]:
        teacher=Image.open(root/f'teacher_{n}px.png') if (root/f'teacher_{n}px.png').exists() else None
        if teacher is None:
            # Teacher full frames are recreated from the already generated
            # comparison contact only for display elsewhere; here use the
            # exact helper result saved below by square_flat_style.
            from square_flat_style import square_flat
            import library as l
            from render_review import rgb_batch
            teacher=Image.fromarray(rgb_batch(square_flat(l.tx(__import__('numpy').array(src)[None]),n))[0])
        student=Image.open(root/f'{n}px/student_outdoor.png').convert('RGB')
        rows.append((f'{n}×{n}教师',teacher,f'{n}×{n}学生',student))
    pixel=Image.open(D/'revisions/film-pixel-r2/review/stills/PIXEL_R39/outdoor_076_new.png').convert('RGB')
    rows.append(('原图',src,'可用像素版：严格4×4',pixel))
    canvas=Image.new('RGB',(1280,len(rows)*520),'#101722');draw=ImageDraw.Draw(canvas)
    for r,(la,a,lb,b) in enumerate(rows):
        y=r*520;draw.text((8,y+4),la,font=font,fill='white');draw.text((648,y+4),lb,font=font,fill='white')
        canvas.paste(a.crop(box).resize(size,Image.Resampling.NEAREST),(0,y+40));canvas.paste(b.crop(box).resize(size,Image.Resampling.NEAREST),(640,y+40))
    canvas.save(root/'方格局部放大对照.png')
    print('SQUARE_ZOOM_REVIEW_COMPLETE')

if __name__=='__main__':main()
