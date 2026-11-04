"""Contact sheets for mean/max/centre/min fixed-S4R2 square students."""
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import json
from library import D,OLD
from render_review import FONT

ROOTS={'平均':D/'experiments/square-flat-v1','最大':D/'experiments/square-max-v1','中心点':D/'experiments/square-center-v1','最小':D/'experiments/square-min-v1'}

def main():
    out=D/'experiments/square-reduction-comparison';out.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype(FONT,18)
    sources=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes']
    # Full four-scene sheet for the only exactly supported cell size.
    canvas=Image.new('RGB',(2400,1200),'#101722');draw=ImageDraw.Draw(canvas)
    for row,s in enumerate(sources):
        n=s['frames']//2;original=Image.open(OLD/'acceptance-four-scenes-5s-20261029/originals'/f"{s['scene']}_{n:03d}.png").convert('RGB')
        pics=[('原图',original)]+[(name,Image.open(root/'4px'/f"student_{s['scene']}.png").convert('RGB')) for name,root in ROOTS.items()]
        for col,(name,pic) in enumerate(pics):
            draw.text((col*480+4,row*300+3),s['label']+' / '+name,font=font,fill='white');canvas.paste(pic.resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
    canvas.save(out/'4x4四种取色_四景.jpg',quality=96)
    # Nearest-neighbour crop makes the actual cells visible.
    box=(800,120,1120,360);cell=(480,360);zoom=Image.new('RGB',(1920,1200),'#101722');zdraw=ImageDraw.Draw(zoom)
    for row,size in enumerate([4,8,16]):
        for col,(name,root) in enumerate(ROOTS.items()):
            p=Image.open(root/f'{size}px/student_outdoor.png').convert('RGB');y=row*400
            metric=json.loads((root/f'{size}px/experiment.json').read_text('utf-8'))
            rng=metric['student_target_square_internal_range'];mae=metric['validation_natural_mae']
            zdraw.text((col*480+4,y+3),f'{size}×{size}目标 / {name} / MAE {mae:.2f} / 格内差 {rng:.2f}',font=font,fill='white')
            zoom.paste(p.crop(box).resize(cell,Image.Resampling.NEAREST),(col*480,y+39))
    zoom.save(out/'三种尺寸四种取色_局部放大.png')
    print('SQUARE_REDUCTION_COMPARISON_COMPLETE')

if __name__=='__main__':main()
