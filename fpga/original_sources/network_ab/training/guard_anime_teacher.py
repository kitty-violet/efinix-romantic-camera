"""Training-only source-edge guidance to suppress false contouring on smooth light."""
import cv2
import numpy as np


def guard_teacher(source, teacher):
    if source.shape!=teacher.shape or source.dtype!=np.uint8 or teacher.dtype!=np.uint8:raise ValueError('Matched RGB images required')
    src=source.astype(np.float32);out=teacher.astype(np.float32)
    gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY);gray=cv2.GaussianBlur(gray,(0,0),1)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    activity=cv2.dilate(np.hypot(gx,gy),np.ones((9,9),np.uint8))
    gate=np.clip((activity-1.5)/4.5,0,1);gate=gate*gate*(3-2*gate)
    gate=cv2.GaussianBlur(gate,(0,0),2)[:,:,None]
    residual=out-src;low=cv2.GaussianBlur(residual,(0,0),10)
    value=src+low+gate*(residual-low)
    return value.round().clip(0,255).astype(np.uint8)
