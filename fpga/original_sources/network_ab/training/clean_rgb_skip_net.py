"""The same 29-node clean-skip graph, with direct RGB instead of a palette.

The three output component convolutions are linearly combined into RGB. No
palette, color quantizer, extra image filter or discriminator runs at inference.
This changes output interpretation and requires a matching RGB program/DMA.
"""
import copy
import torch
from .clean_skip_capacity_net import CleanSkipNet,plan as previous_plan,budget as previous_budget


def plan():
    p=copy.deepcopy(previous_plan());p['output']='signed RGB INT8 plus128; no palette';return p


class CleanRGBNet(CleanSkipNet):
    def __init__(self):super().__init__();self.nodes=plan()['nodes']

    @torch.no_grad()
    def from_components(self,source):
        self.load_state_dict(source.state_dict());c=self.conv('rgb');w=c.weight.clone();b=c.bias.clone()
        transform=w.new_tensor([[1,1,-1],[1,0,1],[1,-1,-1]])
        c.weight.copy_(torch.einsum('ij,jchw->ichw',transform,w));c.bias.copy_(transform@b)

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='clean-rgb-skip-int8-q17-v1',output=plan()['output']);p.pop('levels',None);return p


def budget():
    b=previous_budget();b.update(output='RGB3 no palette',requires_rgb_program=True);return b
