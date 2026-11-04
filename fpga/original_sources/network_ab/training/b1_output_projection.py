"""RGB-coupled L0 projection for TRAINING targets, never B1 inference.

Independent implementation of Xu et al. 2011 half-quadratic equations.
All three colour derivatives share the same support decision. Reflect padding
keeps the FFT's periodic seam outside the supervised central image.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def project(image, strength=.0004, padding=32):
    # Public input/output are signed RGB/128, matching actual B1 tensors.
    z=F.pad((image+1)*128/255,(padding,)*4,mode='reflect')
    original=z.clone();h,w=z.shape[-2:]
    fy=torch.arange(h,device=z.device,dtype=z.dtype)*torch.pi/h
    fx=torch.arange(w//2+1,device=z.device,dtype=z.dtype)*torch.pi/w
    lap=4*fy.sin().square()[:,None]+4*fx.sin().square()[None,:]
    original_fft=torch.fft.rfft2(original)
    beta=2*strength
    while beta<10000:
        dx=torch.roll(z,-1,-1)-z;dy=torch.roll(z,-1,-2)-z
        keep=(dx.square()+dy.square()).sum(1,keepdim=True)>=strength/beta
        dx=dx*keep;dy=dy*keep
        adj=torch.roll(dx,1,-1)-dx+torch.roll(dy,1,-2)-dy
        z=torch.fft.irfft2((original_fft+beta*torch.fft.rfft2(adj))/(1+beta*lap),s=(h,w))
        beta*=2
    z=z[...,padding:-padding,padding:-padding] if padding else z
    return (z*255/128-1).clamp(-1,127/128)


def regression(device):
    z=torch.full((1,3,64,96),.1,device=device)
    assert (project(z)-z).abs().max()<1e-5
    z[...,48:]=.7
    p=project(z)
    assert (p-z).abs().max()<.005
    # Joint support must commute with permutations of RGB.
    torch.manual_seed(8123);z=torch.rand(1,3,64,96,device=device)*1.5-.75
    p=project(z);q=project(z[:,[2,0,1]])
    assert (q-p[:,[2,0,1]]).abs().max()<.0001
    assert torch.isfinite(p).all()
    return dict(constant=True,hard_step=True,shared_rgb_support=True,finite=True)
