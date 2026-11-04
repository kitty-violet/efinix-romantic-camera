"""Shorten colour transitions in the person-only preview; keep TV unchanged."""
import cv2,numpy as np,torch
from .coarse_people import skin_structure,refined_mask,composite
from .coarse_surface_fill import tv_surface,to_tensor,to_array
from .painting_parameters import as_lab,as_rgb,soft_levels

PRESETS={
    'blocks':dict(strength=.10,iterations=100,l_step=12.,ab_step=8.,l_sharp=11.,ab_sharp=8.,l_mix=.96,ab_mix=.88),
    'strongblocks':dict(strength=.10,iterations=100,l_step=15.,ab_step=10.,l_sharp=13.,ab_sharp=10.,l_mix=.98,ab_mix=.92),
}


@torch.inference_mode()
def person_surface(pred,adapter,tone_module,preset='blocks',source=None):
    cfg=PRESETS[preset];skin=skin_structure(pred if source is None else source)
    edge=torch.from_numpy(skin)[None,None].cuda()
    surface=to_array(tv_surface(to_tensor(pred),edge,cfg['strength'],1-.014/cfg['strength'],cfg['iterations']))
    lab=as_lab(tone_module.tone(surface,adapter))
    lab[:,:,1]-=.8*np.clip(np.linalg.norm(lab[:,:,1:],axis=2)/12.,0,1)
    lab[:,:,2]+=.45*np.maximum(-lab[:,:,2]-10.,0)
    # Suppress one-pixel level islands before steepening the colour transitions.
    # The guide respects existing Lab edges; it does not redraw face landmarks.
    base_lab=lab.copy()
    lab=cv2.medianBlur(lab,3)
    lab=cv2.bilateralFilter(lab,9,8.,3.)
    # Keep eye/nose transitions in the colour guide. A coarser body palette
    # must not erase the few pixels that describe a small face.
    lab=lab*(1-.5*skin[:,:,None])+base_lab*(.5*skin[:,:,None])
    # Fixed bins across all people and all frames. Do not vary bin locations
    # with the skin cue: that would introduce extra transitions on the face.
    face=lab.copy()
    face[:,:,0]=soft_levels(face[:,:,0],8.,9.,.84)
    for c in (1,2):face[:,:,c]=soft_levels(face[:,:,c],6.,8.,.75)
    lab[:,:,0]=soft_levels(lab[:,:,0],cfg['l_step'],cfg['l_sharp'],cfg['l_mix'])
    for c in (1,2):lab[:,:,c]=soft_levels(lab[:,:,c],cfg['ab_step'],cfg['ab_sharp'],cfg['ab_mix'])
    return as_rgb(lab*(1-skin[:,:,None])+face*skin[:,:,None])


def main():
    import time
    from PIL import Image,ImageDraw,ImageFont
    from .block_video_runtime import STILLS,load_frozen
    from .runtime import write_json,file_hash
    torch.set_num_threads(2);cv2.setNumThreads(2)
    root=STILLS.parent/'coarse-people-20261005-probe';out=root/'short-transitions-v3';out.mkdir(exist_ok=False)
    _,_,p=load_frozen();adapter=np.load(STILLS/'color_adapter.npz')['lab_chroma_affine'];rows=[]
    for i in (90,210,330,449):
        source=np.array(Image.open(root/f'{i:03d}_source.png'));pred=np.array(Image.open(root/f'{i:03d}_student.png'))
        bg=np.array(Image.open(root/f'{i:03d}_old.png'));mask=refined_mask(np.load(root/f'{i:03d}_person_probability.npy'),source)
        for key in PRESETS:
            start=time.perf_counter();people=person_surface(pred,adapter,p,key,source)
            Image.fromarray(composite(bg,people,mask)).save(out/f'{i:03d}_{key}.png')
            rows.append(dict(frame=i,preset=key,seconds=time.perf_counter()-start))
        print('PERSON_COLOUR_BLOCKS',i,flush=True)
    cases=[(210,(120,435,440,715)),(330,(630,280,950,560)),(330,(0,430,320,710))]
    board=Image.new('RGB',(960,len(cases)*316),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for row,(i,box) in enumerate(cases):
        paths=[(root/'variants-v2'/f'{i:03d}_coarse.png','上一版：过渡较柔'),(out/f'{i:03d}_blocks.png','A：压短颜色过渡'),(out/f'{i:03d}_strongblocks.png','B：更少、更硬的色阶')]
        for col,(path,title) in enumerate(paths):
            draw.text((col*320+4,row*316+4),title,font=font,fill='white');board.paste(Image.open(path).crop(box),(col*320,row*316+34))
    board.save(out/'comparison.png');write_json(out/'experiment.json',dict(presets=PRESETS,timing=rows,code_sha256=file_hash(__file__),tv_unchanged=True,skin_tv_min=.014,no_geometry_warp=True,new_training=False))


if __name__=='__main__':main()
