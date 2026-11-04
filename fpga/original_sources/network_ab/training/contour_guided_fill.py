"""Explicit experimental decoder/fill after actual student RGB prediction.

Not a weights-only FPGA deployment. Canny/component decoding and weighted TV
are real inference stages and must not be attributed to raw neural output.
All seven contour endpoint parameters remain the user's selected parameters.
"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_seven_contours import LEVELS
from .coarse_contour_targets import target_parameters


def contours(predicted_rgb,level):
    return target_parameters(predicted_rgb,LEVELS[level])


def gradient(u):
    dx=F.pad(u[:,:,:,1:]-u[:,:,:,:-1],(0,1,0,0))
    dy=F.pad(u[:,:,1:,:]-u[:,:,:-1,:],(0,0,0,1))
    return dx,dy


def divergence(px,py):
    return px-F.pad(px[:,:,:,:-1],(1,0,0,0))+py-F.pad(py[:,:,:-1,:],(0,0,1,0))


@torch.no_grad()
def fill(rgb,edge,strength=.18,iterations=80):
    x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda().float()/255
    e=torch.from_numpy(edge.copy())[None,None].cuda().float()
    # Small regularization on predicted boundaries, stronger inside regions.
    border=F.max_pool2d(e,3,1,1)
    bound=strength*(1-.92*border)
    u=x.clone();ubar=u.clone();px=torch.zeros_like(x);py=torch.zeros_like(x)
    for _ in range(iterations):
        gx,gy=gradient(ubar);px=px+.5*gx;py=py+.5*gy
        norm=(px.square().sum(1,keepdim=True)+py.square().sum(1,keepdim=True)).sqrt()
        scale=(norm/bound.clamp_min(1e-8)).clamp_min(1);px=px/scale;py=py/scale
        previous=u;u=(u+.125*divergence(px,py)+.125*x)/1.125;ubar=2*u-previous
    return np.uint8((u[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round())
