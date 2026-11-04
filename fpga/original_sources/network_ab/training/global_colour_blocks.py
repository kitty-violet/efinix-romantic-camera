"""Common fixed colour planes for all objects, before display scaling.

Unlike the preceding person-only change, no region retains the old soft colour
curve. This is a software appearance experiment, not an FPGA deployment graph.
"""
import cv2,numpy as np,torch
from .painting_parameters import as_lab,as_rgb,soft_levels
from .coarse_surface_fill import tv_surface,to_tensor,to_array

PRESETS={
    'clear':dict(tv=.012,iterations=48,l_step=14.,ab_step=10.,sharp=18.,mix=.995),
    'strong':dict(tv=.020,iterations=64,l_step=18.,ab_step=12.,sharp=22.,mix=1.),
}


@torch.inference_mode()
def prepare_surface(base,preset='strong'):
    cfg=PRESETS[preset]
    # Suppress weak variations before assigning fixed levels; do not amplify
    # small multiscale contrast and then attempt to flatten it again.
    clean=to_array(tv_surface(to_tensor(base),None,cfg['tv'],0,cfg['iterations']))
    lab=as_lab(clean)
    return lab


def quantize(lab,preset='strong'):
    cfg=PRESETS[preset];q=lab.copy()
    q[:,:,0]=soft_levels(q[:,:,0],cfg['l_step'],cfg['sharp'],cfg['mix'])
    for c in (1,2):q[:,:,c]=soft_levels(q[:,:,c],cfg['ab_step'],cfg['sharp'],cfg['mix'])
    return as_rgb(q)


@torch.inference_mode()
def process(base,preset='strong'):
    return quantize(prepare_surface(base,preset),preset)


def main():
    import json,time
    from PIL import Image,ImageDraw,ImageFont
    from .block_video_runtime import STILLS
    from .runtime import write_json,file_hash
    torch.set_num_threads(2);cv2.setNumThreads(2)
    out=STILLS.parent/'global-colour-blocks-stills-20261005-v2';out.mkdir(exist_ok=False)
    cfg=json.loads((STILLS/'experiment.json').read_text(encoding='utf-8'));records=[]
    for row in cfg['inputs']:
        key=row['id'];base=np.array(Image.open(STILLS/(key+'_balanced.png')))
        assert file_hash(STILLS/(key+'_surface.png'))==row['surface_sha256']
        for name in PRESETS:
            start=time.perf_counter();a=process(base,name)
            Image.fromarray(a).save(out/(key+'_'+name+'.png'))
            records.append(dict(id=key,preset=name,seconds=time.perf_counter()-start))
        print('GLOBAL_COLOUR',key,flush=True)
    cases=[('val_plaza_090',(0,240,420,660)),('val_stairs_090',(120,300,540,720)),('val_new_hallway_120',(0,0,420,420))]
    board=Image.new('RGB',(1680,len(cases)*454),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for row,(key,box) in enumerate(cases):
        paths=[(STILLS/(key+'_teacher.png'),'教师参考'),(STILLS/(key+'_balanced.png'),'之前：背景仍有连续过渡'),(out/(key+'_clear.png'),'全画面：明确色阶'),(out/(key+'_strong.png'),'全画面：更少、更硬的色阶')]
        for col,(path,title) in enumerate(paths):
            draw.text((col*420+6,row*454+4),title,font=font,fill='white');board.paste(Image.open(path).crop(box),(col*420,row*454+34))
    board.save(out/'teacher_comparison.png')
    write_json(out/'experiment.json',dict(presets=PRESETS,records=records,inputs=cfg['inputs'],code_sha256=file_hash(__file__),teacher_only_for_comparison=True,new_training=False,board_validated=False))


if __name__=='__main__':main()
