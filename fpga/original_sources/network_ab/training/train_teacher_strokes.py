"""Teacher's visible dark strokes, not the two sides of a Canny boundary.

Labels are local dark ridges of the SAVED TEACHER RGB. They do not substitute
a filtered source image for the teacher. Morphological closing is training-label
preparation only. Actual inference remains the same compact convolution graph.
"""
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from . import train_native_contours as base
from .train_context_contours import ContextContourNet
from .runtime import file_hash,write_json

LABEL=dict(kind='teacher_dark_ridge',kernel=3,subtract=3.,range=29.,label_only=True,
    note='Teacher local closing minus teacher luma, soft amplitude. Not semantic object masks; not inference morphology.')


def strokes(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    ridge=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)-gray
    return np.clip((ridge-3.)/29.,0,1)


def criterion(logit,target,tracing=.35,margin=16):
    p=logit[...,margin:-margin,margin:-margin].sigmoid();t=target[...,margin:-margin,margin:-margin]
    active=(t>.05).to(t.dtype);background=1-active
    error=F.smooth_l1_loss(p,t,beta=.1,reduction='none')
    ink=(error*active).sum()/active.sum().clamp_min(1)
    quiet=(error*background).sum()/background.sum().clamp_min(1)
    dx=lambda v:v[...,1:]-v[...,:-1]
    dy=lambda v:v[...,1:,:]-v[...,:-1,:]
    gradient=F.l1_loss(dx(p),dx(t))+F.l1_loss(dy(p),dy(t))
    loss=ink+1.5*quiet+2*gradient
    return loss,dict(stroke_amplitude=float(ink.detach()),quiet=float(quiet.detach()),gradient=float(gradient.detach()))


def main():
    base.NativeContourNet=ContextContourNet;base.edges=strokes;base.criterion=criterion;base.LABEL=LABEL
    if '--name' not in sys.argv:sys.argv.extend(['--name','native-teacher-strokes-v1'])
    if sys.argv[1]=='prepare':
        base.main();out=base.ROOT/sys.argv[sys.argv.index('--name')+1]
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
        cfg.update(graph='context90_native_contour',objective='teacher visible dark strokes; amplitude and gradient only',label=LABEL)
        write_json(out/'config.json',cfg);b=json.loads((out/'binding.json').read_text(encoding='utf-8'))
        for p in [Path(__file__),Path(__file__).with_name('train_context_contours.py'),Path(__file__).with_name('train_selective_contours.py')]:
            b['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
        b['config']=file_hash(out/'config.json');write_json(out/'binding.json',b)
        return
    base.main()


if __name__=='__main__':main()
