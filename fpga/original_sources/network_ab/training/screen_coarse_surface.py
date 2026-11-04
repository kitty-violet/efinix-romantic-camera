"""Inspect a scale-first surface against the previous output and teacher."""
import json
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .coarse_surface_fill import process,tv_surface,LEVELS,COLOR
from .train_native_contours import ROOT,load_rgb
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible


def main():
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    out=ROOT/'coarse-surface-color-v3';out.mkdir(exist_ok=False);old=ROOT/'seven-structured-fill-cool-v2/delivery'
    adapter=np.load(old/'color_adapter.npz')['lab_chroma_affine'];data=json.loads((old/'data.json').read_text());records=[]
    x=torch.ones(1,3,64,96,device='cuda')*.37
    for strength in [.025,.055,.085]:
        u=tv_surface(x,None,strength,0,64);assert float((u-x).abs().max())<1e-6
    for r in data['validation']:
        pred=load_rgb(old/(r['id']+'_before.png'))
        for l in ['g1','g4','g7']:
            start=time.perf_counter();rgb,e,stages=process(pred,adapter,l,True)
            Image.fromarray(rgb).save(out/(r['id']+'_'+l+'.png'))
            Image.fromarray(np.uint8((1-e)*255)).save(out/(r['id']+'_'+l+'_edge.png'))
            for name,a in stages.items():Image.fromarray(a).save(out/(r['id']+'_'+l+'_'+name+'.png'))
            records.append(dict(id=r['id'],level=l,seconds=time.perf_counter()-start,edge_fraction=float(e.mean())))
        print('COARSE_SURFACE_SCREEN',r['id'],flush=True)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680)),('val_new_hallway_120',(680,240,1000,520))]
    b=Image.new('RGB',(1600,1264),'#17202a');d=ImageDraw.Draw(b);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for j,(key,box) in enumerate(cases):
        paths=[old/(key+'_teacher.png'),old/(key+'_g4_color.png')]+[out/(key+'_'+l+'.png') for l in ['g1','g4','g7']]
        for i,(title,p) in enumerate(zip(['教师参考','上一版','本次轻档','本次中档','本次较强档'],paths)):
            d.text((320*i+4,316*j+4),title,font=font,fill='white');b.paste(Image.open(p).convert('RGB').crop(box),(320*i,316*j+34))
    b.save(out/'comparison.png')
    write_json(out/'screen.json',dict(records=records,levels=LEVELS,color=COLOR,code_sha256=file_hash(__file__),
        constant_pass=True,board_validated=False,neural_weights_unchanged=True))
    print('COARSE_SURFACE_SCREEN_DONE',flush=True)


if __name__=='__main__':main()
