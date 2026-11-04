"""Training targets for large structures, explicitly excluding fine texture.

Gaussian scale selection and component removal happen on the SAVED TEACHER
only when preparing supervision, never on the actual network prediction.
Lines remain one native pixel wide; coarseness refers to retained structure.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .train_teacher_strokes import strokes
from .runtime import write_json

ROOT=Path('E:/fpga/model-training-workspace/runs/region-detail-deploy-20261004')
LEVELS={
    'a':dict(sigma=2.5,low=24,high=48,min_component_pixels=32,min_extent=14),
    'b':dict(sigma=5.,low=8,high=16,min_component_pixels=48,min_extent=20),
    'c':dict(sigma=8.,low=5,high=10,min_component_pixels=64,min_extent=28),
}


def target(rgb,level):
    return target_parameters(rgb,LEVELS[level])


def target_parameters(rgb,cfg):
    smooth=cv2.GaussianBlur(rgb,(0,0),cfg['sigma'],borderType=cv2.BORDER_REPLICATE)
    edge=cv2.Canny(smooth,cfg['low'],cfg['high'],apertureSize=3,L2gradient=True)
    n,labels,stats,_=cv2.connectedComponentsWithStats(edge,8)
    keep=np.zeros(n,dtype=bool)
    for i in range(1,n):
        _,_,w,h,count=stats[i]
        keep[i]=count>=cfg['min_component_pixels'] and max(w,h)>=cfg['min_extent']
    return keep[labels].astype(np.float32)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='coarse-contour-target-screen-v1');args=parser.parse_args()
    cv2.setNumThreads(2)
    out=ROOT/args.name;out.mkdir(exist_ok=False)
    data=json.loads((ROOT/'native-teacher-strokes-v1/data.json').read_text(encoding='utf-8'))
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    board=Image.new('RGB',(1600,len(cases)*316),'#1b2430');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    rows=[]
    for row in data['validation']:
        rgb=np.array(Image.open(row['target']).convert('RGB'))
        for level in LEVELS:
            t=target(rgb,level);Image.fromarray(np.uint8((1-t)*255)).save(out/(row['id']+'_'+level+'.png'))
            rows.append(dict(id=row['id'],level=level,edge_pixels=int(t.sum()),edge_fraction=float(t.mean())))
    for j,(name,box) in enumerate(cases):
        row=next(r for r in data['validation'] if r['id']==name);rgb=np.array(Image.open(row['target']).convert('RGB'))
        old=Image.fromarray(np.uint8((1-strokes(rgb))*255)).convert('RGB')
        images=[Image.fromarray(rgb),old]+[Image.open(out/(name+'_'+l+'.png')).convert('RGB') for l in LEVELS]
        titles=['教师图','旧：3×3细暗纹目标','A：较简略目标','B：大结构目标','C：更简略目标']
        for i,(im,title) in enumerate(zip(images,titles)):
            d.text((320*i+4,316*j+4),title,font=font,fill='white');board.paste(im.crop(box),(320*i,316*j+34))
    board.save(out/'targets_only.png');write_json(out/'report.json',dict(levels=LEVELS,records=rows,targets_only_not_predictions=True,
        native_line_width=1,task='remove fine structures rather than dilate existing lines',color_training=False))
    print('COARSE_TARGETS_READY',out,flush=True)


if __name__=='__main__':main()
