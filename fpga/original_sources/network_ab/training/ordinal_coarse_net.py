"""Ordered-color head: train three components, fold to the SAME C32 PW.

For each component, argmax(2*c*v-c*c) equals the closest fixed color level.
The factorization exists only during training; exported weights are ordinary
32x8 PW coefficients and use the previously checked FPGA graph unchanged.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from .coarse_skip_net import CoarseSkipNet
from .train_luma_regions import LEVELS


class OrderedWeight(nn.Module):
    def __init__(self):
        super().__init__();cs=[];indices=[];temps=[]
        for i,(levels,temp) in enumerate(zip(LEVELS,[128.,256.,512.])):
            cs.extend(v/128-(1 if i==0 else 0) for v in levels);indices.extend([i]*len(levels));temps.extend([temp]*len(levels))
        self.register_buffer('centers',torch.tensor(cs));self.register_buffer('indices',torch.tensor(indices));self.register_buffer('temperature',torch.tensor(temps))

    def forward(self,w):return (2*self.temperature*self.centers)[:,None,None,None]*w[self.indices]


class OrderedBias(OrderedWeight):
    def forward(self,b):return self.temperature*(2*self.centers*b[self.indices]-self.centers.square())


class OrdinalCoarseNet(CoarseSkipNet):
    def __init__(self,source=None,projection=None):
        super().__init__()
        if source is not None:self.load_state_dict(source)
        head=self.conv('rgb');parametrize.register_parametrization(head,'weight',OrderedWeight());parametrize.register_parametrization(head,'bias',OrderedBias())
        if projection is not None:
            with torch.no_grad():
                head.parametrizations.weight.original.zero_();head.parametrizations.bias.original.zero_()
                head.parametrizations.weight.original[:3,:,0,0].copy_(projection[:8].T)
                head.parametrizations.bias.original[:3].copy_(projection[8])

    def component_prediction(self,features):
        h=self.conv('rgb');return F.conv2d(features,h.parametrizations.weight.original[:3],h.parametrizations.bias.original[:3])

    def folded(self):
        # Do not remove parametrizations on a deepcopy: PyTorch's dynamically
        # generated class may be shared and removal can break the live model.
        h=self.conv('rgb');prefix='layers.rgb.0.'
        ordinary=CoarseSkipNet().to(h.weight.device)
        state={k:v for k,v in self.state_dict().items() if '.parametrizations.' not in k}
        keys=ordinary.state_dict()
        head_prefix=next(k[:-len('weight')] for k in keys if k.startswith('layers.rgb.') and k.endswith('weight'))
        state[head_prefix+'weight']=h.weight.detach().clone()
        state[head_prefix+'bias']=h.bias.detach().clone()
        ordinary.load_state_dict(state);ordinary.scales=copy.deepcopy(self.scales)
        return ordinary.eval()
