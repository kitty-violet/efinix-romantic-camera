"""Source-guided correction of false light patches on low-detail surfaces.

Fixed spatial filter only: no frame statistics, temporal state, material labels,
learned weights, exposure adjustment, or copied face region.
"""
import cv2
import numpy as np


CONFIG={'radius':48,'epsilon_rgb255_squared':64.,'edge_low':1.5,'edge_high':6.,'brightness_low':60.,'brightness_high':140.}


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def correct_surface(source,output,*,radius=48,epsilon=64.):
    if source.shape!=output.shape or source.dtype!=np.uint8 or output.dtype!=np.uint8:raise ValueError('Matched uint8 RGB images required')
    src=source.astype(np.float32);style=output.astype(np.float32)
    gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY);guide=cv2.GaussianBlur(gray,(0,0),1.)
    box=lambda x:cv2.boxFilter(x,-1,(2*radius+1,2*radius+1),normalize=True,borderType=cv2.BORDER_REFLECT_101)
    mi=box(guide);mp=box(style);var=np.maximum(box(guide*guide)-mi*mi,0)
    covariance=box(guide[:,:,None]*style)-mi[:,:,None]*mp
    a=covariance/(var[:,:,None]+epsilon);b=mp-a*mi[:,:,None]
    filtered=box(a)*guide[:,:,None]+box(b)
    gx=cv2.Sobel(guide,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(guide,cv2.CV_32F,0,1,ksize=3)/8
    activity=cv2.dilate(np.hypot(gx,gy),np.ones((9,9),np.uint8))
    edges=cv2.GaussianBlur(smoothstep((activity-1.5)/4.5),(0,0),2.)
    bright=smoothstep((guide-60.)/80.)
    weight=(1-edges)*bright
    result=style+(filtered-style)*weight[:,:,None]
    return result.round().clip(0,255).astype(np.uint8)
