"""Shared teacher with a zero-initialized correction head for full-range output.

No material lookup, scene statistics, temporal filter, or tone LUT at inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .coherent_shared_model import SharedAppearanceNet,conv
except ImportError:  # Standalone exported inference package.
    from coherent_shared_model import SharedAppearanceNet,conv


class SurfaceAppearanceNet(SharedAppearanceNet):
    def __init__(self,width=32,blocks=6,residual_scale=.9):
        super().__init__(width,blocks,residual_scale)
        self.correction=conv(width,3)
        nn.init.zeros_(self.correction[1].weight)
        nn.init.zeros_(self.correction[1].bias)

    def load_previous(self,state):
        result=self.load_state_dict(state,strict=False)
        if set(result.missing_keys)!={'correction.1.weight','correction.1.bias'} or result.unexpected_keys:
            raise ValueError(f'Incompatible previous model: {result}')

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or min(x.shape[-2:])<16:
            raise ValueError('Expected NCHW RGB')
        a=self.stem(x);b=self.down1(a);c=self.body(self.down2(b))
        d=self.up1(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e=self.up2(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return (x+self.residual_scale*torch.tanh(self.head(e))+self.correction(e)).clamp(-1,127/128)
