"""A deliberately small, bounded refinement of the previous appearance.

No TV smoothing, hard palette or removal of broad tonal layers. Slightly shorten
fine colour transitions and steepen existing boundaries without new outlines.
"""
import cv2,numpy as np
from .painting_parameters import as_lab,as_rgb,soft_levels
from .block_separation import steepen

PARAMETERS=dict(light_step=6.,light_sharp=6.,light_mix=.40,
    chroma_step=4.,chroma_sharp=5.,chroma_mix=.24,boundary_mix=.50,
    max_lab_change=[1.8,1.,1.])


def process(rgb, cfg=None):
    cfg=PARAMETERS if cfg is None else cfg
    original=as_lab(rgb);lab=original.copy()
    lab[:,:,0]=soft_levels(lab[:,:,0],cfg['light_step'],cfg['light_sharp'],cfg['light_mix'])
    for c in (1,2):lab[:,:,c]=soft_levels(lab[:,:,c],cfg['chroma_step'],cfg['chroma_sharp'],cfg['chroma_mix'])
    lab+=cfg['boundary_mix']*(steepen(lab,1)-lab)
    limit=np.array(cfg['max_lab_change'],np.float32)
    lab=original+np.clip(lab-original,-limit,limit)
    return as_rgb(lab)


def main():
    import json,shutil
    from pathlib import Path
    from PIL import Image,ImageDraw,ImageFont
    from .block_video_runtime import STILLS
    from .fpga_video_resize import bilinear_2to3
    from .runtime import file_hash,write_json
    cv2.setNumThreads(2)
    out=STILLS.parent/'gentle-global-boundaries-20261005-v2';out.mkdir(exist_ok=False)
    native=STILLS.parent/'global-colour-blocks-video-20261005-v2'
    records=[]
    cases=[('plaza_090',native/'preflight_090_before720.png',(0,240,420,660)),
           ('plaza_330',native/'preflight_330_before720.png',(630,280,1050,700)),
           ('stairs_090',STILLS/'val_stairs_090_balanced.png',(120,300,540,720)),
           ('hallway_120',STILLS/'val_new_hallway_120_balanced.png',(0,0,420,420))]
    board=Image.new('RGB',(840,len(cases)*454),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for row,(name,path,box) in enumerate(cases):
        before=np.array(Image.open(path));after=process(before);error=np.abs(after.astype(float)-before.astype(float))
        Image.fromarray(before).save(out/(name+'_before720.png'));Image.fromarray(after).save(out/(name+'_after720.png'))
        Image.fromarray(bilinear_2to3(after)).save(out/(name+'_1080p.png'))
        for col,(value,title) in enumerate([(before,'之前的画面'),(after,'小幅调整：只收紧过渡与边界')]):
            draw.text((col*420+5,row*454+4),title,font=font,fill='white');board.paste(Image.fromarray(value).crop(box),(col*420,row*454+34))
        records.append(dict(id=name,input=str(path),input_sha256=file_hash(path),mae=float(error.mean()),max=int(error.max()),p95=float(np.percentile(error,95)),fraction_over8=float((error.max(2)>8).mean())))
    board.save(out/'comparison.png')
    board.crop((0,0,840,908)).save(out/'trees_people_comparison.png')
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'experiment.json',dict(parameters=PARAMETERS,records=records,code_sha256=file_hash(__file__),hard_palette=False,extra_smoothing=False,new_training=False))
    print(json.dumps(records,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
