"""Matched tree / grass / person ablations at native resolution."""
from pathlib import Path
import json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .block_separation import transform_surface,PRESETS
from .painting_parameters import as_lab
from .runtime import configure_workspace,write_json,file_hash

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OLD=ROOT/'directional-paint-20261004-v3'
OUT=ROOT/'block-separation-20261004-v2'


def load(p):return np.array(Image.open(p).convert('RGB'))


def main():
    configure_workspace();cv2.setNumThreads(2);OUT.mkdir(exist_ok=False)
    records=[]
    for key in ['val_plaza_090','val_stairs_090','val_new_hallway_120']:
        base=load(OLD/(key+'_surface.png'))
        for name in PRESETS:
            start=time.perf_counter();out=transform_surface(base,name)
            Image.fromarray(out).save(OUT/(key+'_'+name+'.png'))
            records.append(dict(id=key,preset=name,seconds=time.perf_counter()-start))
        print('BLOCK_SCREEN',key,flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cases=[('val_plaza_090',(10,280,330,560)),('val_plaza_090',(80,400,400,680)),
           ('val_stairs_090',(140,380,460,660)),('val_new_hallway_120',(0,0,320,280))]
    sheet=Image.new('RGB',(1280,316*len(cases)),'#17202a');d=ImageDraw.Draw(sheet)
    for j,(key,box) in enumerate(cases):
        columns=[(OLD/(key+'_teacher.png'),'教师'),(OLD/(key+'_direction4.png'),'上一版')]
        columns += [(OUT/(key+'_'+n+'.png'),title) for n,title in [('balanced','块面分明'),('cleanblocks','更少颜色层次')]]
        for i,(path,title) in enumerate(columns):
            d.text((i*320+5,j*316+5),title,font=font,fill='white')
            sheet.paste(Image.open(path).crop(box),(i*320,j*316+34))
    sheet.save(OUT/'screen.png')
    # Same-location diagnostics: teacher is used only here, never by the transform.
    boxes={'lit_leaf':[26,289,76,322],'dark_tree':[55,396,100,440],'blue_crown':[108,246,166,281]}
    stats={}
    for suffix in ['teacher','student','tone','surface','direction4']:
        lab=as_lab(load(OLD/('val_plaza_090_'+suffix+'.png')))
        stats[suffix]={}
        for name,(x0,y0,x1,y1) in boxes.items():
            q=lab[y0:y1,x0:x1];stats[suffix][name]=dict(mean_lab=q.mean((0,1)).tolist(),chroma_mean=float(np.linalg.norm(q[:,:,1:],axis=2).mean()))
        tree=lab[280:560,10:330]
        stats[suffix]['tree_L_std']=float(tree[:,:,0].std())
    write_json(OUT/'screen.json',dict(records=records,presets=PRESETS,diagnostic_boxes=boxes,diagnostic_stats=stats,
        base_files={k:file_hash(OLD/(k+'_surface.png')) for k in ['val_plaza_090','val_stairs_090','val_new_hallway_120']}))
    print('BLOCK_SCREEN_DONE',flush=True)


if __name__=='__main__':main()
