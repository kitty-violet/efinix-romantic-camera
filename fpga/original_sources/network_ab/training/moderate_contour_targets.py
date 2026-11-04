"""Small coarseness increments BETWEEN the saved fine style and former A."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .coarse_contour_targets import target_parameters,target as former_target,ROOT
from .train_teacher_strokes import strokes
from .runtime import write_json

LEVELS={
    'light':dict(sigma=.65,low=36,high=72,min_component_pixels=12,min_extent=6),
    'middle':dict(sigma=1.05,low=32,high=64,min_component_pixels=16,min_extent=8),
    'strong':dict(sigma=1.5,low=28,high=56,min_component_pixels=24,min_extent=10),
}


def target(rgb):
    return np.stack([target_parameters(rgb,v) for v in LEVELS.values()],axis=-1).astype(np.uint8)


def main():
    cv2.setNumThreads(2);out=ROOT/'moderate-contour-target-screen-v1';out.mkdir(exist_ok=False)
    data=json.loads((ROOT/'native-teacher-strokes-v1/data.json').read_text(encoding='utf-8'))
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    board=Image.new('RGB',(1920,len(cases)*316),'#1b2430');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);rows=[]
    for row in data['validation']:
        rgb=np.array(Image.open(row['target']).convert('RGB'));maps=target(rgb)
        for k,name in enumerate(LEVELS):
            Image.fromarray(np.uint8((1-maps[:,:,k])*255)).save(out/(row['id']+'_'+name+'.png'))
            rows.append(dict(id=row['id'],level=name,edge_pixels=int(maps[:,:,k].sum())))
    for j,(name,box) in enumerate(cases):
        row=next(r for r in data['validation'] if r['id']==name);rgb=np.array(Image.open(row['target']).convert('RGB'))
        images=[Image.fromarray(rgb),Image.fromarray(np.uint8((1-strokes(rgb))*255)).convert('RGB')]
        images += [Image.open(out/(name+'_'+l+'.png')).convert('RGB') for l in LEVELS]
        images += [Image.fromarray(np.uint8((1-former_target(rgb,'a'))*255)).convert('RGB')]
        titles=['教师图','旧细纹目标','轻减细节','中间档','较强减细节','之前A（仅作上界对照）']
        for i,(im,title) in enumerate(zip(images,titles)):
            d.text((320*i+4,316*j+4),title,font=font,fill='white');board.paste(im.crop(box),(320*i,316*j+34))
    board.save(out/'targets_only.png');write_json(out/'report.json',dict(levels=LEVELS,records=rows,targets_only_not_predictions=True,
        user_range='between old and former A; former B/C rejected; no coloring until user accepts predictions',native_line_width=1))
    print('MODERATE_TARGETS_READY',out,flush=True)


if __name__=='__main__':main()
