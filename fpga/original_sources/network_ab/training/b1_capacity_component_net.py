"""Spend the B1 resident-parameter budget on low-resolution region features.

The shared B1 contract is not modified. Same DW/PW/ReLU/nearest vocabulary,
but this is a different graph requiring separately compiled descriptors.
"""
import copy
import torch
from torch import nn
from .component_region_net import ComponentNet,plan as small_plan


def plan():
    wide32={'down.pw','context.down.dw','context.down.pw','wide.down.dw','wide.down.pw','res1_conv1.dw','res1_conv1.pw','res1_conv2.dw','res1_conv2.pw','res1_add','decode.dw'}
    wide16={'decode.pw','wide.up','wide.up.dw','wide.up.pw','context.up','context.up.dw'}
    nodes=[];shapes={'input':[720,1280,3]}
    for old in small_plan()['nodes']:
        n=copy.deepcopy(old)
        if n['id']=='res1_conv1.dw':n['inputs']=['capacity.mix2']
        if n['id']=='res1_add':n['inputs'][1]='capacity.mix2'
        ci=shapes[n['inputs'][0]][2];co=32 if n['id'] in wide32 else 16 if n['id'] in wide16 else n['channels']
        n.update(input_channels=ci,channels=co,input_shape_hwc=shapes[n['inputs'][0]].copy())
        n['output_shape_hwc'][2]=co;n['groups']=ci if n['kind']=='dw3x3' else 1
        n['terms']=ci*n['kernel']**2//n['groups'] if n['kernel'] else 0
        n['macs']=n['output_shape_hwc'][0]*n['output_shape_hwc'][1]*co*n['terms'];nodes.append(n);shapes[n['id']]=n['output_shape_hwc']
        if n['id']=='wide.down.pw':
            for name,parent in [('capacity.mix1','wide.down.pw'),('capacity.mix2','capacity.mix1')]:
                q=copy.deepcopy(n);q.update(id=name,inputs=[parent],input_channels=32,input_shape_hwc=[90,160,32],terms=32,macs=90*160*32*32)
                nodes.append(q);shapes[name]=q['output_shape_hwc']
    result=dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False,output=small_plan()['output'])
    assert len(nodes)==29 and sum(n['kernel']>0 for n in nodes)==24
    assert sum(n['channels']*n['terms'] for n in nodes)<=10256
    assert result['macs']<=966758400
    assert {n['terms'] for n in nodes if n['kernel']}<={8,9,16,27,32}
    return result


class CapacityComponentNet(ComponentNet):
    def __init__(self):
        nn.Module.__init__(self);self.nodes=plan()['nodes'];self.scales=None;self.layers=nn.ModuleDict()
        for n in self.nodes:
            if not n['kernel']:continue
            parts=[nn.ReplicationPad2d(1)] if n['kernel']==3 else []
            c=nn.Conv2d(n['input_channels'],n['channels'],n['kernel'],n['stride'],groups=n['groups']);parts.append(c)
            if n['activation']=='relu':parts.append(nn.ReLU())
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)
        self.eval()

    @torch.no_grad()
    def widen_from(self,source):
        generator=torch.Generator().manual_seed(2026100454)
        for n in self.nodes:
            if not n['kernel']:continue
            c=self.conv(n['id'])
            if n['id'].startswith('capacity.'):
                c.weight.zero_();c.weight[:,:,0,0].copy_(torch.eye(c.in_channels));c.bias.zero_();continue
            old=source.conv(n['id']);oi=torch.arange(c.out_channels)%old.out_channels
            if n['kind']=='dw3x3':weight=old.weight[oi].clone()
            else:
                assert c.in_channels%old.in_channels==0
                copies=c.in_channels//old.in_channels;ii=torch.arange(c.in_channels)%old.in_channels
                weight=old.weight[oi][:,ii].clone()/copies
                if copies>1:
                    # Different zero-sum outgoing shares break duplicate-channel
                    # gradient symmetry while preserving the initial function.
                    shape=(c.out_channels,copies,old.in_channels,*c.kernel_size)
                    noise=torch.randn(shape,generator=generator)*old.weight.std().clamp_min(.01)*.03
                    noise-=noise.mean(1,keepdim=True);weight+=noise.reshape_as(weight)
            c.weight.copy_(weight);c.bias.copy_(old.bias[oi])

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='b1-capacity-component-int8-q17-v1');return p


def budget():
    nodes=plan()['nodes'];weight=1792;bias=12608;mult=15040;relu=17472;rows=[]
    align=lambda x:(x+63)//64*64
    for n in nodes:
        if not n['kernel']:continue
        pad=(n['channels']+7)//8*8;weight=align(weight);bias=align(bias);mult=align(mult)
        rows.append(dict(id=n['id'],weight_offset=weight,bias_offset=bias,multiplier_offset=mult,relu_offset=relu,padded_channels=pad))
        weight+=pad*n['terms'];bias+=pad*4;mult+=pad*4;relu+=pad//8
    assert weight<=12608 and bias<=15040 and mult<=17472 and relu<=17600,(weight,bias,mult,relu)
    return dict(raw_weights=sum(n['channels']*n['terms'] for n in nodes),trainable_parameters=sum(p.numel() for p in CapacityComponentNet().parameters()),
        ideal_macs=plan()['macs'],nodes=len(nodes),convolutions=len(rows),resident_capacity=20480,valid_image_bytes=17792,
        ends=dict(weight=weight,bias=bias,multiplier=mult,relu=relu),operators=rows,fpga_fit=False,compiled_program=False)
