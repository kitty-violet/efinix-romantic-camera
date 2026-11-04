"""Hard global colour planes with fixed shadow-sensitive luminance levels."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from . import publish_global_blocks as base
from .painting_parameters import as_rgb
from .runtime import file_hash,write_json

OUT=base.ROOT/'global-colour-blocks-video-20261005-v2'
LIGHT=np.array([0,6,12,20,29,40,53,68,84,96],np.float32)
CHROMA_STEP=6.


def colour_planes(lab,previous=None,flow=None):
    unit=lab.copy();unit[:,:,0]=np.interp(lab[:,:,0],LIGHT,np.arange(len(LIGHT)))
    unit[:,:,1:]/=CHROMA_STEP
    bins=np.floor(unit+.5);held=np.zeros(lab.shape[:2],bool)
    if previous is not None:
        mx,my,valid=flow
        prior=cv2.remap(previous['bins'],mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
        guide=cv2.remap(previous['lab'],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        reliable=valid&(np.abs(lab-guide).max(2)<4.)
        keep=reliable[:,:,None]&(np.abs(unit-prior)<.56)
        held=np.any(keep&(bins!=prior),2);bins=np.where(keep,prior,bins)
    out=bins.copy();out[:,:,0]=LIGHT[np.clip(bins[:,:,0],0,len(LIGHT)-1).astype(np.int32)]
    out[:,:,1:]*=CHROMA_STEP;out[out[:,:,0]==0,1:]=0
    return as_rgb(out),bins.astype(np.float32),held


def main(mode):
    base.OUT=OUT;base.colour_planes=colour_planes
    if mode!='preflight':
        cfg=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'))
        assert cfg['code'][str(Path(__file__))]==file_hash(__file__)
    base.run(mode)
    if mode=='preflight':
        cfg=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'))
        cfg['code'][str(Path(__file__))]=file_hash(__file__)
        cfg['colour_steps']=dict(lightness=LIGHT.tolist(),a=CHROMA_STEP,b=CHROMA_STEP)
        cfg['reason']='All objects use hard shared levels; narrower low-light spacing preserves structure without restoring continuous gradients. Exact black has neutral chroma.'
        shutil.copy2(__file__,OUT/'source_code'/Path(__file__).name);write_json(OUT/'pipeline.json',cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','short','render']);main(parser.parse_args().mode)
