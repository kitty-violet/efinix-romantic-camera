"""A supervised structure bottleneck followed by a learned RGB renderer.

Every path to RGB goes through structure.pw. Its eight nonnegative channels
represent teacher region RGB, four signed directional contrasts, and color
boundary magnitude. Targets exist only during training, never inference.
Same DW/PW/ReLU/nearest/add vocabulary, new descriptor graph, not board proven.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .clean_rgb_skip_net import CleanRGBNet, plan as previous_plan


def plan():
    old = previous_plan(); nodes = []
    templates = {n['id']: n for n in old['nodes']}
    for row in old['nodes']:
        n = copy.deepcopy(row)
        if n['id'].startswith('native.texture.'): continue
        if n['id'] == 'context.skip_add': n['inputs'][1] = 'native.pw'
        if n['id'] == 'up_nn': n['inputs'] = ['structure.pw']
        nodes.append(n)
        if n['id'] == 'context.skip_add':
            for suffix, parent in [('dw', 'context.skip_add'), ('pw', 'structure.dw')]:
                q = copy.deepcopy(templates['native.texture.' + suffix])
                q.update(id='structure.' + suffix, inputs=[parent]); nodes.append(q)
    return dict(nodes=nodes, macs=sum(n['macs'] for n in nodes), production=False,
                output='signed RGB INT8 plus128; no palette',
                bottleneck='structure.pw', bypass_to_rgb=False,
                structure_channels=['region_R','region_G','region_B','dx_positive',
                                    'dx_negative','dy_positive','dy_negative','color_boundary'])


def scaffold_target(rgb):
    """Teacher-derived training label; gradients are NOT source-image edges."""
    base = F.avg_pool2d((rgb + 1) * .5, 2)
    dx = F.pad(base[..., 1:] - base[..., :-1], (0,1,0,0))
    dy = F.pad(base[..., 1:, :] - base[..., :-1, :], (0,0,0,1))
    gx, gy = dx.mean(1, keepdim=True), dy.mean(1, keepdim=True)
    magnitude = torch.maximum(dx.abs().amax(1, keepdim=True), dy.abs().amax(1, keepdim=True))
    edges = torch.cat([F.relu(gx),F.relu(-gx),F.relu(gy),F.relu(-gy),magnitude],1)
    return torch.cat([base, (edges*4).clamp(0,1)],1)


class ContourRegionNet(CleanRGBNet):
    def __init__(self):
        nn.Module.__init__(self); self.nodes=plan()['nodes']; self.scales=None; self.layers=nn.ModuleDict()
        for n in self.nodes:
            if not n['kernel']: continue
            parts=[nn.ReplicationPad2d(1)] if n['kernel']==3 else []
            parts.append(nn.Conv2d(n['input_channels'], n['channels'], n['kernel'], n['stride'], groups=n['groups']))
            if n['activation']=='relu': parts.append(nn.ReLU())
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)

    def encode(self, x):
        values={'input':x}
        for n in self.nodes:
            a=values[n['inputs'][0]]
            if n['kind']=='nearest_x2': v=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            elif n['kind']=='residual_add': v=a+values[n['inputs'][1]]
            else: v=self.layers[n['id'].replace('.','__')](a)
            values[n['id']]=v
            if n['id']=='structure.pw': return v
        raise AssertionError('missing structure bottleneck')

    def render(self, scaffold):
        v=scaffold; started=False
        for n in self.nodes:
            if n['id']=='structure.pw': started=True; continue
            if not started: continue
            assert len(n['inputs'])==1, 'A source bypass would defeat this experiment'
            if n['kind']=='nearest_x2': v=v.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                v=self.layers[n['id'].replace('.','__')](v)
                if n['activation']=='clamp_rgb': v=v.clamp(-1,127/128)
        return v

    @torch.no_grad()
    def initialize(self, source):
        for name in self.layers:
            if name in source.layers: self.layers[name].load_state_dict(source.layers[name].state_dict())
        for name in ('structure.dw','up.dw','finish.dw'):
            c=self.conv(name); c.weight.zero_(); c.weight[:,0,1,1]=1; c.bias.zero_()
        c=self.conv('structure.pw'); nn.init.normal_(c.weight,std=.01); c.bias[:3]=.4; c.bias[3:]=.02
        c=self.conv('mix.pw'); c.weight.zero_(); c.weight[:,:,0,0].copy_(torch.eye(8)); c.bias.zero_()
        c=self.conv('rgb'); c.weight.zero_(); c.bias.fill_(-1)
        for i in range(3): c.weight[i,i,0,0]=2

    def integer_package(self):
        p=super().integer_package();p.update(plan=plan(),schema='contour-region-int8-q17-v1');return p


def budget():
    nodes=plan()['nodes']; weight=1792;bias=12608;mult=15040;relu=17472; rows=[]
    align=lambda x:(x+63)//64*64
    for n in nodes:
        if not n['kernel']: continue
        pad=(n['channels']+7)//8*8;weight=align(weight);bias=align(bias);mult=align(mult)
        rows.append(dict(id=n['id'],weight_offset=weight,bias_offset=bias,multiplier_offset=mult,relu_offset=relu,padded_channels=pad))
        weight+=pad*n['terms'];bias+=pad*4;mult+=pad*4;relu+=pad//8
    assert weight<=12608 and bias<=15040 and mult<=17472 and relu<=17600
    assert plan()['macs']<=966758400
    return dict(raw_weights=sum(n['channels']*n['terms'] for n in nodes),trainable_parameters=sum(p.numel() for p in ContourRegionNet().parameters()),
        ideal_macs=plan()['macs'],nodes=len(nodes),convolutions=len(rows),resident_capacity=20480,valid_image_bytes=17792,
        ends=dict(weight=weight,bias=bias,multiplier=mult,relu=relu),operators=rows,fpga_fit=False,compiled_program=False)
