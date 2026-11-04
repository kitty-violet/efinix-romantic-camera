"""Image-space anisotropic sector filtering; research approximation, not RTL.

Implements structure-tensor-aligned ellipse sectors and variance-weighted color
means, following the method family of Kyprianidis et al. 2009. Discrete sampling
and rendering parameters are ours, not an exact port of the authors' code.
"""
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .painting_parameters import tone, as_lab, as_rgb, soft_levels

PRESETS={
    'direction4':dict(label='E 方向色块·较轻',radius=4.,tensor_sigma=2.,variance_scale=.025,variance_power=8,
                      light_step=7.,light_mix=.40,edge_strength=12.),
    'direction6':dict(label='F 方向色块·中档',radius=6.,tensor_sigma=2.,variance_scale=.025,variance_power=8,
                      light_step=7.,light_mix=.40,edge_strength=12.),
}


def sector_samples(radius):
    # Smooth sector characteristic functions, then Gaussian radial weighting.
    # At maximum elongation, adjacent samples remain <=1 native pixel apart.
    # Sparse 9x9 quadrature aliases fine grass into artificial grain.
    extent=max(4,int(math.ceil(2*radius)))
    y,x=np.mgrid[-extent:extent+1,-extent:extent+1].astype(np.float32)/extent
    angle=np.arctan2(y,x);radial=np.exp(-2*(x*x+y*y));kernels=[]
    for i in range(8):
        d=np.arctan2(np.sin(angle-i*math.pi/4),np.cos(angle-i*math.pi/4))
        k=(np.abs(d)<math.pi/8).astype(np.float32);k[extent,extent]=1/8
        kernels.append(cv2.GaussianBlur(k,(0,0),.1625*extent)*radial)
    points=[];weights=[]
    for iy in range(2*extent+1):
        for ix in range(2*extent+1):
            if x[iy,ix]**2+y[iy,ix]**2<=1.001:
                points.append([x[iy,ix],y[iy,ix]])
                weights.append([k[iy,ix] for k in kernels])
    weights=np.array(weights,np.float32).T
    weights/=weights.sum(axis=1,keepdims=True)
    return np.array(points,np.float32),weights


def geometry(rgb,sigma):
    x=rgb.astype(np.float32)/255
    smooth=cv2.GaussianBlur(x,(0,0),.8)
    dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    e,f,g=[cv2.GaussianBlur(z,(0,0),sigma) for z in [(dx*dx).sum(2),(dx*dy).sum(2),(dy*dy).sum(2)]]
    anis=np.sqrt((e-g)**2+4*f*f)/(e+g+1e-8)
    # Major ellipse axis follows the tangent (minimum variation), not the normal.
    angle=.5*np.arctan2(2*f,e-g)+np.pi/2
    return np.cos(angle).astype(np.float32),np.sin(angle).astype(np.float32),anis.astype(np.float32)


@torch.no_grad()
def surface(rgb,cfg,return_geometry=False):
    h,w=rgb.shape[:2];cos,sin,anis=geometry(rgb,cfg['tensor_sigma'])
    device='cuda';x=torch.from_numpy(rgb.copy()).permute(2,0,1).to(device).float()[None]/255
    c,s,a=[torch.from_numpy(v).to(device) for v in (cos,sin,anis)]
    scale=1+a;major=cfg['radius']*scale;minor=cfg['radius']/scale
    yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    coords,weight=sector_samples(cfg['radius']);weight=torch.from_numpy(weight).to(device)
    mean=torch.zeros((8,3,h,w),device=device);sq=torch.zeros_like(mean)
    for start in range(0,len(coords),7):
        p=torch.from_numpy(coords[start:start+7]).to(device);count=len(p)
        u=p[:,0,None,None]*major;v=p[:,1,None,None]*minor
        gx=(xx+u*c-v*s)*(2/(w-1))-1;gy=(yy+u*s+v*c)*(2/(h-1))-1
        grid=torch.stack((gx,gy),-1)
        sampled=F.grid_sample(x.expand(count,-1,-1,-1),grid,mode='bilinear',padding_mode='border',align_corners=True)
        weights=weight[:,start:start+count]
        mean+=torch.einsum('kn,nchw->kchw',weights,sampled)
        sq+=torch.einsum('kn,nchw->kchw',weights,sampled.square())
    variance=(sq-mean.square()).clamp_min(0).mean(1,keepdim=True)
    logratio=.5*torch.log(variance.clamp_min(1e-12))-math.log(cfg['variance_scale'])
    alpha=torch.softmax(-F.softplus(cfg['variance_power']*logratio),dim=0)
    result=(alpha*mean).sum(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    out=np.uint8(np.round(result*255))
    if return_geometry:return out,dict(anisotropy=anis,cos=cos,sin=sin)
    return out


def render(base,cfg):
    lab=as_lab(base);l=lab[:,:,0].copy()
    lab[:,:,0]=soft_levels(l,cfg['light_step'],3.,cfg['light_mix'])
    # Modest coarse dark-edge contrast; no original fine-detail residual is added.
    narrow=cv2.GaussianBlur(l,(0,0),.9);broad=cv2.GaussianBlur(l,(0,0),1.8)
    response=np.maximum(broad-narrow-.55,0)/2.5
    ink=1-np.exp(-response**2)
    lab[:,:,0]=np.clip(lab[:,:,0]-cfg['edge_strength']*ink,0,100)
    return as_rgb(lab),np.uint8(np.round((1-ink)*255))


def transform(rgb,adapter,preset='direction6',stages=False):
    cfg=PRESETS[preset];colored=tone(rgb,adapter)
    base=surface(colored,cfg);out,edges=render(base,cfg)
    if stages:return out,dict(tone=colored,surface=base,edges=edges)
    return out
