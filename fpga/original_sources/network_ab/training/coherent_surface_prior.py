"""Training-only, spatially shared tone/edge-aware surface repair targets.

An explicit simplification prior, not proof of game-material synthesis.
"""
import math
import torch
from torch.nn import functional as F


def luminance(x):
    return (x*x.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)


def bilateral_surface(image,guide,iterations=2,sigma=.055):
    """Positive, normalized local averaging; strong guide edges receive low weight.

    Both image and guide use RGB 0..1. Weights depend on spatial content only.
    """
    h,w=image.shape[-2:];gpad=F.pad(guide,(2,)*4,mode='reflect')
    weights=[]
    for dy in range(-2,3):
        for dx in range(-2,3):
            neighbor=gpad[:,:,2+dy:2+dy+h,2+dx:2+dx+w]
            diff=(neighbor-guide).square().mean(1,keepdim=True)
            weights.append((dy,dx,torch.exp(-diff/(2*sigma*sigma)-(dx*dx+dy*dy)/8)))
    denom=sum(v for _,_,v in weights)
    result=image
    for _ in range(iterations):
        pad=F.pad(result,(2,)*4,mode='reflect')
        result=sum(weight*pad[:,:,2+dy:2+dy+h,2+dx:2+dx+w]
                   for dy,dx,weight in weights)/denom
    return result


def apply_tone_prior(x,prior):
    """Fixed global parameters fitted only to training frames and the previous teacher."""
    lum=luminance(x).clamp(0,1)
    knots=x.new_tensor(prior['luma_knots'])
    position=lum*(len(knots)-1);lo=position.floor().long().clamp(max=len(knots)-2)
    t=position-lo
    mapped=knots[lo]*(1-t)+knots[lo+1]*t
    features=torch.cat((x[:,0:1]-lum,x[:,2:3]-lum,mapped),1)
    matrix=x.new_tensor(prior['chroma_matrix'])
    ch=torch.einsum('nchw,cd->ndhw',features,matrix)
    ch=ch-luminance(ch)
    return (mapped+ch).clamp(0,1)


@torch.no_grad()
def repair_target(source,previous,prior,mode,strength=.8):
    x=(source+1)*128/255;old=(previous+1)*128/255
    if mode=='light':target=apply_tone_prior(x,prior)
    elif mode=='texture':
        target=old+strength*(bilateral_surface(old,x)-old)
    elif mode=='combined':
        surface=x+strength*(bilateral_surface(x,x)-x)
        target=apply_tone_prior(surface,prior)
    else:raise ValueError(mode)
    return target*255/128-1
