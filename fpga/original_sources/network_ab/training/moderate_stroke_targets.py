"""Gently remove weak isolated teacher strokes without changing their positions.

Offline teacher labels only. The old soft dark-ridge definition and its thin
line positions stay unchanged. No Canny sides, thickening or inference filter.
"""
import cv2
import numpy as np
from .train_teacher_strokes import strokes

LEVELS = {
    'light': dict(threshold=.12, min_pixels=4, min_extent=4),
    'middle': dict(threshold=.20, min_pixels=6, min_extent=6),
    'strong': dict(threshold=.28, min_pixels=8, min_extent=8),
}


def target(rgb):
    old=strokes(rgb); outputs=[]
    for cfg in LEVELS.values():
        count,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(old>cfg['threshold']),8)
        keep=np.zeros(count,dtype=bool)
        for i in range(1,count):
            _,_,w,h,n=stats[i]
            keep[i]=n>=cfg['min_pixels'] and max(w,h)>=cfg['min_extent']
        outputs.append(old*keep[labels])
    return np.stack(outputs,axis=-1).astype(np.float32)
