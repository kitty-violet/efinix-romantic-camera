"""RGB region student using the existing hardware operator vocabulary.

Graph changes require a new program/parameter package. No argmax, palette,
image-dependent normalization, nonlinear filter or temporal state at inference.
"""
import copy
from fractions import Fraction
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .region_budget_net import RegionNet, plan as base_plan


def plan():
    p=copy.deepcopy(base_plan());nodes=p['nodes'];i=next(i for i,n in enumerate(nodes) if n['id']=='finish.dw')
    pw=copy.deepcopy(next(n for n in nodes if n['id']=='mix.pw'))
    pw.update(id='finish.pw',inputs=['mix_add'],activation='relu')
    nodes[i]['inputs']=['finish.pw'];nodes.insert(i,pw)
    p.update(macs=sum(n['macs'] for n in nodes),production=False,output='signed RGB int8 plus128')
    return p


def round_away(x):return x.sign()*(x.abs()+.5).floor()
def ste_round(x):return round_away(x).detach()+(x-x.detach())
def ratio(x):
    f=Fraction(float(x));return [f.numerator,f.denominator]


class DetailRGBNet(RegionNet):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes'];self.scales=None
        c=nn.Conv2d(8,8,1)
        with torch.no_grad():c.weight.zero_();c.bias.zero_();c.weight[:,:,0,0].copy_(torch.eye(8))
        self.layers['finish__pw']=nn.Sequential(c,nn.ReLU())

    def quant_parameters(self,n):
        layer=self.conv(n['id']);si=self.scales[n['inputs'][0]];so=self.scales[n['id']]
        w,b=layer.weight,layer.bias
        # Bound bias to exact FP32 integer range, below hardware's INT32 limit.
        sw=w.detach().abs().amax((1,2,3)).clamp_min(1e-9)/127
        sw=torch.maximum(sw,b.detach().abs()/(si*2**22))
        sw=sw.clamp_min(so/(si*65536))
        q=ste_round(w/sw[:,None,None,None]).clamp(-127,127)
        qb=ste_round(b/(si*sw))
        mult=round_away((si*sw/so).double()*131072).clamp(1,131071).detach()
        return q,qb,mult,sw

    def forward(self,x,return_nodes=False):
        if self.scales is None:
            if not return_nodes:return super().forward(x)
            values={'input':x}
            for n in self.nodes:
                a=values[n['inputs'][0]]
                if n['kind']=='nearest_x2':v=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
                elif n['kind']=='residual_add':v=a+values[n['inputs'][1]]
                else:
                    v=self.layers[n['id'].replace('.','__')](a)
                    if n['activation']=='clamp_rgb':v=v.clamp(-1,127/128)
                values[n['id']]=v
            return values
        values={'input':ste_round(x/self.scales['input']).clamp(-128,127)*self.scales['input']}
        for n in self.nodes:
            name=n['id'];a=values[n['inputs'][0]];si=self.scales[n['inputs'][0]];so=self.scales[name]
            if n['kind']=='nearest_x2':v=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            elif n['kind']=='residual_add':
                # Both branches calibrated to precisely the same scale.
                assert si==self.scales[n['inputs'][1]]
                acc=a/si+values[n['inputs'][1]]/si
                m=int(np.clip(np.floor(si/so*131072+.5),1,131071))
                v=ste_round(acc.double()*m/131072).clamp(-128,127).float()*so
            else:
                qw,qb,m,sw=self.quant_parameters(n)
                ai=ste_round(a/si)
                if n['kernel']==3:ai=F.pad(ai,(1,)*4,mode='replicate')
                acc=F.conv2d(ai,qw,stride=n['stride'],groups=n['groups'])
                acc=acc.double()+qb.double()[None,:,None,None]
                v=ste_round(acc*m[None,:,None,None]/131072).clamp(-128,127).float()*so
                if n['activation']=='relu':v=v.clamp_min(0)
            values[name]=v
        return values if return_nodes else values['rgb']

    @torch.no_grad()
    def calibrate(self,inputs):
        self.scales=None;maxima={}
        for x in inputs:
            values=self.forward(x,True)
            for name,v in values.items():maxima[name]=max(maxima.get(name,0),float(v.abs().max()))
        forced={n['inputs'][0]:n['inputs'][1] for n in self.nodes if n['kind']=='residual_add'}
        scales={'input':1/128}
        for n in self.nodes:
            name=n['id'];si=scales[n['inputs'][0]]
            if n['kind']=='nearest_x2':so=si
            elif name in forced:so=scales[forced[name]]
            elif n['kind']=='residual_add':so=max(maxima[name]*1.08/127,si*1.001)
            elif n['activation']=='clamp_rgb':so=1/128
            else:so=max(maxima[name]*1.08/127,1/512)
            scales[name]=so
        self.scales=scales
        return dict(scales=scales,maxima=maxima,scheme='fixed activations, shared residual branch scale, per-OC int8 weights, Q17 half-away')

    @torch.no_grad()
    def integer_package(self):
        assert self.scales is not None
        parameters={};residual={}
        for n in self.nodes:
            name=n['id'];si=self.scales[n['inputs'][0]];so=self.scales[name]
            if n['kind']=='residual_add':
                residual[name]=dict(main_scaled=True,multiplier=131071,
                    post_multiplier=int(np.clip(np.floor(si/so*131072+.5),1,131071)),
                    parameters=dict(main_scale=ratio(si),skip_scale=ratio(si),common_scale=ratio(si),output_scale=ratio(so)))
            elif n['kernel']:
                qw,qb,m,sw=self.quant_parameters(n)
                assert float(qb.abs().max())<2**31
                parameters[name]=dict(weight_oihw=qw.to(torch.int64).cpu().tolist(),bias_int32=qb.to(torch.int64).cpu().tolist(),
                    multiplier_q17=m.to(torch.int64).cpu().tolist(),input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(v) for v in sw.cpu().tolist()])
        return dict(schema='detail-rgb-int8-q17-v1',plan=plan(),parameters=parameters,residual=residual,scales=self.scales,
                    output='signed int8 RGB, add128',production=False,shared_hardware_changed=False)


def fused_plan():
    nodes=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=nodes.pop();assert p['id']==n['inputs'][0]
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],
                input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        nodes.append(n)
    nodes[-1]['id']='rgb_head'
    return dict(nodes=nodes)
