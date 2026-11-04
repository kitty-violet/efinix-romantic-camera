"""Exact inactive-anchor skipping for the approved still pipeline's video render."""
import importlib
import sys
import types
from pathlib import Path
import cv2
import numpy as np

STILLS=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004\block-separation-20261004-v3')


def load_frozen():
    name='frozen_block_video'
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(STILLS/'source_code')];sys.modules[name]=package
    block=importlib.import_module(name+'.block_separation')
    direction=importlib.import_module(name+'.directional_paint')
    paint=importlib.import_module(name+'.painting_parameters')
    return block,direction,paint


def local_contrast_exact(x,sigma,amount,bounds,anchor_step=4.):
    # Same float32 calculations and active-anchor order as the frozen version.
    # Only omit anchors whose interpolation weights are zero at every used level.
    g=[x]
    for _ in range(6):g.append(cv2.pyrDown(g[-1]))
    def up(v,shape):return cv2.pyrUp(v,dstsize=(shape[1],shape[0]))
    original=[g[i]-up(g[i+1],g[i].shape) for i in range(6)]
    adjusted={i:np.zeros_like(original[i]) for i in range(1,5)}
    ranges={i:(float(g[i].min()),float(g[i].max())) for i in range(1,5)}
    strength=[0.,.65,1.,.85,.40,0.]
    for anchor in np.arange(bounds[0],bounds[1]+anchor_step*.5,anchor_step):
        active=[i for i,(low,high) in ranges.items() if anchor>low-anchor_step and anchor<high+anchor_step]
        if not active:continue
        delta=x-anchor;m=[x+amount*delta*np.exp(-.5*(delta/sigma)**2)]
        for _ in range(max(active)+1):m.append(cv2.pyrDown(m[-1]))
        for i in active:
            weight=np.maximum(1-np.abs(g[i]-anchor)/anchor_step,0)
            adjusted[i]+=weight*(m[i]-up(m[i+1],m[i].shape))
    out=g[-1]
    for i in reversed(range(6)):
        band=original[i] if i in (0,5) else original[i]+strength[i]*(adjusted[i]-original[i])
        out=up(out,band.shape)+band
    return out
