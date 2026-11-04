"""Person-only colour-plane abstraction on actual RGB student predictions.

The existing background recipe is untouched. TV removes small internal changes
without coordinate warps; fixed, continuous colour levels avoid per-frame palettes.
This is an offline prototype, not an FPGA-validated operator.
"""
import cv2
import numpy as np
import torch
from .coarse_surface_fill import tv_surface,to_tensor,to_array
from .painting_parameters import as_lab,as_rgb,soft_levels

PRESETS={
    'gentle':dict(strength=.060,iterations=80,l_step=9.,ab_step=6.,mix=.80),
    'coarse':dict(strength=.100,iterations=100,l_step=11.,ab_step=7.,mix=.87),
    'strong':dict(strength=.145,iterations=120,l_step=13.,ab_step=8.,mix=.90),
}


def refined_mask_reference(probability,source):
    # Joint bilateral refinement snaps the low-resolution semantic mask to the
    # source's local colour boundaries. No face detector or hand-painted crop.
    guide=cv2.cvtColor(source,cv2.COLOR_RGB2Lab).astype(np.float32)
    h,w=probability.shape;acc=np.zeros_like(probability);weight=np.zeros_like(probability)
    pp=cv2.copyMakeBorder(probability,8,8,8,8,cv2.BORDER_REFLECT_101)
    gp=cv2.copyMakeBorder(guide,8,8,8,8,cv2.BORDER_REFLECT_101)
    for oy in range(-8,9,2):
        for ox in range(-8,9,2):
            other=gp[8+oy:8+oy+h,8+ox:8+ox+w]
            z=np.exp(-((guide-other)**2).sum(2)/(2*16.**2)-(ox*ox+oy*oy)/(2*5.**2))
            acc+=z*pp[8+oy:8+oy+h,8+ox:8+ox+w];weight+=z
    p=acc/np.maximum(weight,1e-7)
    x=np.clip((p-.12)/.7,0,1)
    return x*x*(3-2*x)


@torch.inference_mode()
def refined_mask(probability,source):
    # Same 81 neighbours, coefficients and accumulation order as the CPU
    # reference. Accelerate the calculation, not the effect or its resolution.
    guide=cv2.cvtColor(source,cv2.COLOR_RGB2Lab).astype(np.float32)
    h,w=probability.shape
    gp=torch.from_numpy(cv2.copyMakeBorder(guide,8,8,8,8,cv2.BORDER_REFLECT_101)).cuda()
    pp=torch.from_numpy(cv2.copyMakeBorder(probability,8,8,8,8,cv2.BORDER_REFLECT_101)).cuda()
    center=gp[8:8+h,8:8+w];acc=torch.zeros_like(pp[8:8+h,8:8+w]);weight=torch.zeros_like(acc)
    for oy in range(-8,9,2):
        for ox in range(-8,9,2):
            other=gp[8+oy:8+oy+h,8+ox:8+ox+w]
            delta=(center-other).square()
            distance=(delta[:,:,0]+delta[:,:,1])+delta[:,:,2]
            z=torch.exp(-distance/(2*16.**2)-(ox*ox+oy*oy)/(2*5.**2))
            acc+=z*pp[8+oy:8+oy+h,8+ox:8+ox+w];weight+=z
    p=acc/weight.clamp_min(1e-7);x=((p-.12)/.7).clamp(0,1)
    return (x*x*(3-2*x)).cpu().numpy()


@torch.inference_mode()
def skin_structure(source):
    ycc=cv2.cvtColor(source,cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr,cb=ycc[:,:,1],ycc[:,:,2]
    ramp=lambda x:np.clip(x,0,1)
    skin=ramp((cr-131)/9)*ramp((182-cr)/15)*ramp((cb-78)/15)*ramp((133-cb)/12)
    # Continuous chroma cue only controls smoothing strength, never generates
    # colours or displaces anatomy. Widen to include eyes inside skin regions.
    skin=cv2.dilate(skin,np.ones((7,7),np.uint8))
    return cv2.GaussianBlur(skin,(0,0),1.4)


@torch.inference_mode()
def person_surface(pred,adapter,tone_module,preset='coarse',source=None):
    cfg=PRESETS[preset]
    # Work before the direction filter, contrast amplification and edge ink:
    # those stages can make a tiny face or a wrinkle into a large jagged region.
    skin=skin_structure(pred if source is None else source)
    edge=torch.from_numpy(skin)[None,None].cuda()
    surface=to_array(tv_surface(to_tensor(pred),edge,cfg['strength'],1-.014/cfg['strength'],cfg['iterations']))
    lab=as_lab(tone_module.tone(surface,adapter))
    lab[:,:,1]-=.8*np.clip(np.linalg.norm(lab[:,:,1:],axis=2)/12.,0,1)
    lab[:,:,2]+=.45*np.maximum(-lab[:,:,2]-10.,0)
    lab[:,:,0]=soft_levels(lab[:,:,0],cfg['l_step']*(1-skin)+7*skin,5.,cfg['mix']*(1-skin)+.55*skin)
    for c in (1,2):lab[:,:,c]=soft_levels(lab[:,:,c],cfg['ab_step'],4.,.55)
    return as_rgb(lab)


def composite(background,people,mask):
    return np.uint8(np.round(background.astype(np.float32)*(1-mask[:,:,None])+people.astype(np.float32)*mask[:,:,None]))


def main():
    import json,time
    from pathlib import Path
    from PIL import Image,ImageDraw,ImageFont
    from .block_video_runtime import load_frozen,STILLS
    from .runtime import write_json
    torch.set_num_threads(2);cv2.setNumThreads(2)
    root=STILLS.parent/'coarse-people-20261005-probe';out=root/'variants-v2';out.mkdir(exist_ok=False)
    _,_,p=load_frozen();adapter=np.load(STILLS/'color_adapter.npz')['lab_chroma_affine'];timing=[]
    for frame in (90,210,330,449):
        load=lambda name:np.array(Image.open(root/f'{frame:03d}_{name}.png'))
        pred,old,source=load('student'),load('old'),load('source')
        start=time.perf_counter();mask=refined_mask(np.load(root/f'{frame:03d}_person_probability.npy'),source)
        Image.fromarray(np.uint8(mask*255)).save(out/f'{frame:03d}_mask.png')
        for key in PRESETS:
            people=person_surface(pred,adapter,p,key,source);result=composite(old,people,mask)
            Image.fromarray(result).save(out/f'{frame:03d}_{key}.png')
        timing.append(dict(frame=frame,seconds=time.perf_counter()-start));print('PEOPLE_VARIANTS',timing[-1],flush=True)
    cases=[(210,(120,435,440,715)),(330,(630,280,950,560)),(330,(0,430,320,710))]
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    board=Image.new('RGB',(1600,316*len(cases)),'#17202a');draw=ImageDraw.Draw(board)
    for row,(i,box) in enumerate(cases):
        entries=[(root/f'{i:03d}_source.png','原片'),(root/f'{i:03d}_old.png','上一版'),(out/f'{i:03d}_gentle.png','A 适度'),(out/f'{i:03d}_coarse.png','B 更粗'),(out/f'{i:03d}_strong.png','C 强')]
        for col,(path,title) in enumerate(entries):
            draw.text((col*320+4,row*316+4),title,font=f,fill='white');board.paste(Image.open(path).crop(box),(col*320,row*316+34))
    board.save(out/'comparison.png');write_json(out/'timing.json',timing)


if __name__=='__main__':main()
