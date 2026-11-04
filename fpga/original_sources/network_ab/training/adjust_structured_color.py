"""Small contour spacing change plus an explicit learned color adapter."""
import cv2
import numpy as np
from .coarse_contour_targets import target_parameters
from .train_seven_contours import LEVELS


def contours(rgb,level):
    params=dict(LEVELS[level]);params['sigma']+=.06
    return target_parameters(rgb,params)


def adapt_rgb(rgb,coefficient,strength=1.):
    x=rgb.astype(np.float32)/255
    y=x@coefficient[:3]+coefficient[3]
    return np.uint8((np.clip(x+strength*(y-x),0,1)*255).round())


def adapt_lab(rgb,coefficient,chroma_gain=1.):
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)
    x=lab/100;chroma=(x@coefficient[:3]+coefficient[3])*100*chroma_gain
    lab[:,:,1:]=chroma
    y=cv2.cvtColor(lab,cv2.COLOR_Lab2RGB)
    return np.uint8((np.clip(y,0,1)*255).round())
