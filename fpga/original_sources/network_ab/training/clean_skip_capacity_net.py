"""Relocate two low-resolution PW layers to the texture-carrying 360p skip.

Same operator set and 29 descriptors. This is a NEW program graph and needs
independent allocation/RTL checks. No inference-time diagnostic box filter.
"""
import copy
import torch
from torch import nn
from .b1_capacity_component_net import plan as old_plan,CapacityComponentNet


def plan():
    nodes=[]
    for row in old_plan()['nodes']:
        n=copy.deepcopy(row)
        if n['id'].startswith('capacity.'):continue
        n['inputs']=['wide.down.pw' if x=='capacity.mix2' else 'native.texture.pw' if x=='native.pw' else x for x in n['inputs']]
        nodes.append(n)
        if n['id']=='native.pw':
            for name,parent,kind,k in [('native.texture.dw','native.pw','dw3x3',3),('native.texture.pw','native.texture.dw','pw1x1',1)]:
                q=copy.deepcopy(n);q.update(id=name,inputs=[parent],kind=kind,kernel=k,groups=8 if k==3 else 1,terms=9 if k==3 else 8,activation='relu')
                q['macs']=360*640*8*q['terms'];nodes.append(q)
    assert len(nodes)==29
    return dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False,output=old_plan()['output'])


class CleanSkipNet(CapacityComponentNet):
    def __init__(self):
        nn.Module.__init__(self);self.nodes=plan()['nodes'];self.scales=None;self.layers=nn.ModuleDict()
        for n in self.nodes:
            if not n['kernel']:continue
            parts=[nn.ReplicationPad2d(1)] if n['kernel']==3 else []
            parts.append(nn.Conv2d(n['input_channels'],n['channels'],n['kernel'],n['stride'],groups=n['groups']))
            if n['activation']=='relu':parts.append(nn.ReLU())
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)

    @torch.no_grad()
    def initialize(self,source):
        for n in self.nodes:
            if not n['kernel']:continue
            c=self.conv(n['id'])
            if n['id'].startswith('native.texture.'):
                c.weight.zero_();c.bias.zero_()
                if n['kernel']==3:c.weight[:,0,1,1]=1
                else:c.weight[:,:,0,0].copy_(torch.eye(8))
            else:c.load_state_dict(source.conv(n['id']).state_dict())

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='clean-skip-capacity-int8-q17-v1');return p


def budget():
    nodes=plan()['nodes'];weight=1792;bias=12608;mult=15040;relu=17472;rows=[]
    align=lambda x:(x+63)//64*64
    for n in nodes:
        if not n['kernel']:continue
        pad=(n['channels']+7)//8*8;weight=align(weight);bias=align(bias);mult=align(mult)
        rows.append(dict(id=n['id'],weight_offset=weight,bias_offset=bias,multiplier_offset=mult,relu_offset=relu,padded_channels=pad))
        weight+=pad*n['terms'];bias+=pad*4;mult+=pad*4;relu+=pad//8
    assert weight<=12608 and bias<=15040 and mult<=17472 and relu<=17600
    assert plan()['macs']<=966758400
    return dict(raw_weights=sum(n['channels']*n['terms'] for n in nodes),trainable_parameters=sum(p.numel() for p in CleanSkipNet().parameters()),
        ideal_macs=plan()['macs'],nodes=len(nodes),convolutions=len(rows),resident_capacity=20480,valid_image_bytes=17792,
        ends=dict(weight=weight,bias=bias,multiplier=mult,relu=relu),operators=rows,fpga_fit=False,compiled_program=False)
