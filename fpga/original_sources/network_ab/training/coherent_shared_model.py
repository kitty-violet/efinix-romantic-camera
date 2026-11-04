"""Shared local residual appearance model; no frame-dependent normalization.

Experimental teacher only. It does not modify the FPGA model contract.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv(a,b,k=3):
    return nn.Sequential(nn.ReflectionPad2d(k//2),nn.Conv2d(a,b,k))


class ResidualBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.a=conv(c,c); self.b=conv(c,c)
    def forward(self,x):
        return x+0.2*self.b(F.leaky_relu(self.a(x),.1))


class BlurDown(nn.Module):
    def __init__(self,c):
        super().__init__()
        kernel=torch.tensor([1.,2.,1.]); kernel=kernel[:,None]*kernel[None,:]/16
        self.register_buffer('kernel',kernel[None,None].repeat(c,1,1,1))
        self.c=c
    def forward(self,x):
        return F.conv2d(F.pad(x,(1,1,1,1),mode='reflect'),self.kernel,stride=2,groups=self.c)


class SharedAppearanceNet(nn.Module):
    def __init__(self,width=24,blocks=6,residual_scale=.9):
        super().__init__()
        self.config={'width':width,'blocks':blocks,'residual_scale':residual_scale}
        self.stem=nn.Sequential(conv(3,width,5),nn.LeakyReLU(.1),ResidualBlock(width))
        self.down1=nn.Sequential(BlurDown(width),conv(width,width*2),nn.LeakyReLU(.1))
        self.down2=nn.Sequential(BlurDown(width*2),conv(width*2,width*4),nn.LeakyReLU(.1))
        self.body=nn.Sequential(*(ResidualBlock(width*4) for _ in range(blocks)))
        self.up1=nn.Sequential(conv(width*6,width*2),nn.LeakyReLU(.1),ResidualBlock(width*2))
        self.up2=nn.Sequential(conv(width*3,width),nn.LeakyReLU(.1),ResidualBlock(width))
        self.head=conv(width,3)
        nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)
        self.residual_scale=residual_scale

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or min(x.shape[-2:])<16:
            raise ValueError('Expected NCHW RGB image')
        a=self.stem(x); b=self.down1(a); c=self.body(self.down2(b))
        d=self.up1(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e=self.up2(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return (x+self.residual_scale*torch.tanh(self.head(e))).clamp(-1,127/128)


def warp(image,backward):
    n,_,h,w=image.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=image.device,dtype=image.dtype),
                         torch.arange(w,device=image.device,dtype=image.dtype),indexing='ij')
    gx=2*(xx[None]+backward[:,0])/max(w-1,1)-1
    gy=2*(yy[None]+backward[:,1])/max(h-1,1)-1
    return F.grid_sample(image,torch.stack((gx,gy),-1),mode='bilinear',
                         padding_mode='border',align_corners=True)


def temporal_residual_loss(current,next_frame,current_out,next_out,flow,mask):
    # Match appearance changes, allowing the source's real motion/exposure change.
    residual=(next_out-next_frame)-warp(current_out-current,flow)
    return (residual.abs()*mask).sum()/(mask.sum()*3+1e-6)
