"""Bounded, full-resolution still test of local-direction color abstraction."""
import ast,json,time,shutil
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .directional_paint import transform,surface,PRESETS
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OUT=ROOT/'directional-paint-20261004-v2';OLD=ROOT/'painting-parameters-20261004-v2'


def main():
    configure_workspace();set_reproducible(2026100492);cv2.setNumThreads(2)
    OUT.mkdir(exist_ok=False)
    for name in ['directional_paint.py',Path(__file__).name]:ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    checks={}
    for key,cfg in PRESETS.items():
        flat=np.full((64,96,3),127,np.uint8);constant=surface(flat,cfg)
        err=int(np.abs(constant.astype(int)-flat).max());assert err==0
        step=flat.copy();step[:,:48]=40;step[:,48:]=200;y=surface(step,cfg)[32,:,0]
        width=int(((y>65)&(y<175)).sum());assert width<=2
        checks[key]=dict(constant_error=err,step_transition_width=width)
    adapter=np.load(OLD/'color_adapter.npz')['lab_chroma_affine'];records=[]
    for key in ['val_stairs_090','val_plaza_090']:
        rgb=np.array(Image.open(OLD/(key+'_student.png')))
        for preset in PRESETS:
            start=time.perf_counter();result,stages=transform(rgb,adapter,preset,True)
            Image.fromarray(result).save(OUT/(key+'_'+preset+'.png'))
            for name,a in stages.items():Image.fromarray(a).save(OUT/(key+'_'+preset+'_'+name+'.png'))
            records.append(dict(id=key,preset=preset,seconds=time.perf_counter()-start))
        print('DIRECTIONAL_SCREEN',key,flush=True)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    b=Image.new('RGB',(1600,948),'#17202a');d=ImageDraw.Draw(b);f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for j,(key,box) in enumerate(cases):
        images=[(OLD/(key+'_teacher.png'),'教师'),(OLD/(key+'_poster.png'),'旧B'),(OLD/(key+'_poster_coarse.png'),'旧C')]
        images += [(OUT/(key+'_'+p+'.png'),PRESETS[p]['label']) for p in PRESETS]
        for i,(path,title) in enumerate(images):
            d.text((i*320+4,j*316+4),title,font=f,fill='white');b.paste(Image.open(path).crop(box),(i*320,j*316+34))
    b.save(OUT/'screen_comparison.png')
    write_json(OUT/'screen.json',dict(presets=PRESETS,checks=checks,records=records,board_validated=False,neural_training=False,
        code={name:file_hash(Path(__file__).with_name(name)) for name in ['directional_paint.py',Path(__file__).name]}))
    print('DIRECTIONAL_SCREEN_DONE',flush=True)


if __name__=='__main__':main()
