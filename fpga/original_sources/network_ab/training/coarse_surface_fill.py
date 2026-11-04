"""Scale-first cartoon surface: remove fine fidelity targets before outlining.

Research pipeline after the real learned RGB prediction; not B1 deployment.
No texture/noise is painted on top. No per-frame exposure statistics or palette
bins are used. The old protected raw-detail path is deliberately absent.
"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .contour_guided_fill import gradient,divergence
from .coarse_contour_targets import target_parameters

LEVELS={f'g{i+1}':dict(pre_strength=.025+.01*i,fill_strength=.12+.005*i,
    sigma=1.15+.10*i,low=32,high=64,min_component_pixels=18+2*i,min_extent=8,
    protection=.70,pre_iterations=64,fill_iterations=96) for i in range(7)}
COLOR=dict(chroma_gain=1.23,cool_b=.85,green_a=.35,black_l=.014,l_gamma=1.035)


def to_tensor(a):return torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/255
def to_array(t):return np.uint8((t[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round())


@torch.no_grad()
def tv_surface(x,edge,strength,protection,iterations):
    if edge is None:bound=torch.full_like(x[:,:1],strength)
    else:
        border=F.max_pool2d(edge,3,1,1);bound=strength*(1-protection*border)
    u=x.clone();extrap=u.clone();px=torch.zeros_like(x);py=torch.zeros_like(x)
    for _ in range(iterations):
        gx,gy=gradient(extrap);px+=.5*gx;py+=.5*gy
        norm=(px.square().sum(1,keepdim=True)+py.square().sum(1,keepdim=True)).sqrt()
        scale=(norm/bound).clamp_min(1);px/=scale;py/=scale
        previous=u;u=(u+.125*divergence(px,py)+.125*x)/1.125;extrap=2*u-previous
    return u


def color_finish(rgb,adapter):
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)
    chroma=((lab/100)@adapter[:3]+adapter[3])*100*COLOR['chroma_gain']
    gate=np.clip(np.linalg.norm(chroma,axis=2)/10,0,1)
    chroma[:,:,0]-=COLOR['green_a']*gate
    chroma[:,:,1]-=COLOR['cool_b']*gate
    light=np.clip((lab[:,:,0]/100-COLOR['black_l'])/(1-COLOR['black_l']),0,1)**COLOR['l_gamma']
    lab[:,:,0]=light*100;lab[:,:,1:]=chroma
    out=cv2.cvtColor(lab,cv2.COLOR_Lab2RGB)
    return np.uint8((np.clip(out,0,1)*255).round())


@torch.no_grad()
def process(rgb,adapter,level='g4',return_stages=False):
    cfg=LEVELS[level];x=to_tensor(rgb)
    guide=tv_surface(x,None,cfg['pre_strength'],0,cfg['pre_iterations'])
    guide_rgb=to_array(guide)
    params={k:cfg[k] for k in ['sigma','low','high','min_component_pixels','min_extent']}
    edge=target_parameters(guide_rgb,params);e=torch.from_numpy(edge.copy())[None,None].cuda()
    # Both the contour and the data term come from the simplified surface.
    # Returning to raw RGB here would restore the rejected small texture.
    merged=tv_surface(guide,e,cfg['fill_strength'],cfg['protection'],cfg['fill_iterations'])
    surface=to_array(merged);colored=color_finish(surface,adapter)
    if return_stages:return colored,edge,dict(guide=guide_rgb,surface=surface)
    return colored,edge
