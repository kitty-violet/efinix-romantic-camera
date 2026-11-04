"""Fixed image-space block separation on the already simplified neural output.

Sampled local-Laplacian remapping and bounded, colour-coupled shock-like
steepening. This is a project approximation, not either paper's exact code.
No original RGB detail, material masks, teacher pixels, or frame histogram.
"""
import cv2
import numpy as np
from .painting_parameters import as_lab, as_rgb, soft_levels

PRESETS = {
    'balanced': dict(contrast=.4, steps=2, l_step=6., mix=.84, ab_step=4., ink=1.),
    'cleanblocks': dict(contrast=.4, steps=2, l_step=8., mix=.88, ab_step=5., ink=1.),
}


def pyramid(x, levels=6):
    result = [x]
    for _ in range(levels):
        result.append(cv2.pyrDown(result[-1]))
    return result


def up(x, shape):
    return cv2.pyrUp(x, dstsize=(shape[1], shape[0]))


def local_contrast(x, sigma, amount, bounds, anchor_step=4.):
    """Fixed-anchor interpolation, change 2--16px bands; leave finest unchanged."""
    g = pyramid(x)
    original = [g[i] - up(g[i+1], g[i].shape) for i in range(len(g)-1)]
    adjusted = [np.zeros_like(v) for v in original]
    # Continuous remap around an anchor, tending to identity for large edges.
    # No 1px residual amplification; large lighting gradients also stay unchanged.
    strength = [0., .65, 1., .85, .40, 0.]
    for anchor in np.arange(bounds[0], bounds[1]+anchor_step*.5, anchor_step):
        delta = x-anchor
        mapped = x + amount * delta * np.exp(-.5*(delta/sigma)**2)
        m = pyramid(mapped)
        for i in range(len(original)):
            weight = np.maximum(1-np.abs(g[i]-anchor)/anchor_step, 0)
            adjusted[i] += weight * (m[i]-up(m[i+1], m[i].shape))
    out = g[-1]
    for i in reversed(range(len(original))):
        band = original[i] + strength[i]*(adjusted[i]-original[i])
        out = up(out, band.shape)+band
    return out


def steepen(lab, steps):
    """Move together toward an existing neighbour, never invent a channel extreme."""
    x = lab.copy()
    for _ in range(steps):
        l = x[:,:,0]
        guide = cv2.GaussianBlur(l, (0,0), 1.15)
        dx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        dy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        norm = np.sqrt(dx*dx+dy*dy)+1e-6
        nx, ny = dx/norm, dy/norm
        dxx = cv2.Sobel(guide, cv2.CV_32F, 2, 0, ksize=3, scale=1/4)
        dyy = cv2.Sobel(guide, cv2.CV_32F, 0, 2, ksize=3, scale=1/4)
        dxy = cv2.Sobel(guide, cv2.CV_32F, 1, 1, ksize=3, scale=1/4)
        second = nx*nx*dxx + 2*nx*ny*dxy + ny*ny*dyy
        lo=x.copy(); hi=x.copy()
        p = cv2.copyMakeBorder(x,1,1,1,1,cv2.BORDER_REFLECT_101)
        h,w=l.shape
        for oy,ox in [(0,1),(1,0),(1,2),(2,1)]:
            q=p[oy:oy+h,ox:ox+w]
            lo=np.where((q[:,:,0]<lo[:,:,0])[:,:,None],q,lo)
            hi=np.where((q[:,:,0]>hi[:,:,0])[:,:,None],q,hi)
        chosen=np.where((second>0)[:,:,None],lo,hi)
        # Continuous weak-gradient gate and bounded update, identical for all scenes.
        span=hi[:,:,0]-lo[:,:,0]
        strong_edge_gate=np.exp(-(span/12.)**4)
        gate=(norm*norm/(norm*norm+.15**2))*np.tanh(np.abs(second)/.15)*strong_edge_gate
        x += (.36*gate)[:,:,None]*(chosen-x)
    return x


def block_activity(lab):
    """Avoid quantization bands on sky/wall ramps using local direction diversity."""
    smooth=cv2.GaussianBlur(lab,(0,0),1.2)
    dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    e,f,g=[cv2.GaussianBlur(z,(0,0),5.) for z in [(dx*dx).sum(2),(dx*dy).sum(2),(dy*dy).sum(2)]]
    coherence=np.sqrt((e-g)**2+4*f*f)/(e+g+1e-7)
    magnitude=np.sqrt(e+g)
    gate=np.clip(2.5*(1-coherence)*magnitude/(magnitude+.2),0,1)
    return cv2.GaussianBlur(gate,(0,0),2.)


def transform_surface(base, preset='balanced', stages=False):
    cfg=PRESETS[preset];lab=as_lab(base);details={}
    activity=block_activity(lab)
    details['activity']=np.uint8(np.round(activity*255))
    # Previous global cooling overshot blue already; do not boost saturation again.
    chroma=np.linalg.norm(lab[:,:,1:],axis=2)
    lab[:,:,1]-=.8*np.clip(chroma/12.,0,1)
    lab[:,:,2]+=.45*np.maximum(-lab[:,:,2]-10.,0)
    if cfg['contrast']:
        lab[:,:,0]=local_contrast(lab[:,:,0], 8., 1.05*cfg['contrast'], (0,100))
        for channel in (1,2):
            lab[:,:,channel]=local_contrast(lab[:,:,channel], 7., .65*cfg['contrast'], (-128,128))
        lab[:,:,0]=np.clip(lab[:,:,0],0,100)
    details['contrast']=as_rgb(lab)
    lab[:,:,0]=soft_levels(lab[:,:,0],cfg['l_step'],8.,cfg['mix']*activity)
    for channel in (1,2):
        lab[:,:,channel]=soft_levels(lab[:,:,channel],cfg['ab_step'],6.,.65*activity)
    details['blocks']=as_rgb(lab)
    lab=steepen(lab,cfg['steps']);details['boundaries']=as_rgb(lab)
    l=lab[:,:,0].copy()
    narrow=cv2.GaussianBlur(l,(0,0),.9);broad=cv2.GaussianBlur(l,(0,0),1.8)
    response=np.maximum(broad-narrow-.55,0)/2.5
    ink=1-np.exp(-response**2)
    # Additional weak dark-side separation belongs only to simplified boundaries.
    weak=np.maximum(broad-narrow-.16,0)/1.8
    lab[:,:,0]=np.clip(lab[:,:,0]-12*ink-cfg['ink']*(1-np.exp(-weak**2)),0,100)
    result=as_rgb(lab)
    if stages:return result,details
    return result
