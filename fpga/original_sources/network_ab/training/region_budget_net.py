"""Independent one-downsample region student. Existing op vocabulary, new graph.

Keeps native boundary features and mixes them BEFORE a nonlinear DW layer.
No normalization from image statistics, RGB output paste, algorithm, or temporal
state at inference. This is not the production B1 contract or coefficient ABI.
"""
import copy
import torch
from torch import nn
from network_ab.model_contract import compile_contract,load_contract


def plan():
    old={n['id']:n for n in compile_contract(load_contract())['nodes']};nodes=[]
    def add(name,kind,parents,h,ci,co,stride=1,activation='relu'):
        k=3 if kind in ('conv3x3','dw3x3') else 1 if kind=='pw1x1' else 0
        out_h=h//stride if kind!='nearest_x2' else h*2
        groups=ci if kind=='dw3x3' else 1
        nodes.append(dict(id=name,kind=kind,inputs=parents,input_channels=ci,channels=co,stride=stride,
            input_shape_hwc=[h,h*16//9,ci],output_shape_hwc=[out_h,out_h*16//9,co],
            kernel=k,groups=groups,activation=activation,terms=ci*k*k//groups if k else 0,
            macs=out_h*(out_h*16//9)*co*ci*k*k//groups if k else 0))
    add('stem','conv3x3',['input'],720,3,8)
    add('down.dw','dw3x3',['stem'],720,8,8,2)
    add('down.pw','pw1x1',['down.dw'],360,8,16)
    p='down.pw'
    for b in (1,2):
        skip=p
        for layer in (1,2):
            name=f'res{b}_conv{layer}'
            add(name+'.dw','dw3x3',[p],360,16,16);p=name+'.dw'
            add(name+'.pw','pw1x1',[p],360,16,16,activation='none' if layer==2 else 'relu');p=name+'.pw'
        add(f'res{b}_add','residual_add',[p,skip],360,16,16,activation='none');p=f'res{b}_add'
    add('decode.dw','dw3x3',[p],360,16,16)
    add('decode.pw','pw1x1',['decode.dw'],360,16,8)
    add('up_nn','nearest_x2',['decode.pw'],360,8,8,activation='none')
    add('up.dw','dw3x3',['up_nn'],720,8,8)
    add('mix.pw','pw1x1',['up.dw'],720,8,8,activation='none')
    add('mix_add','residual_add',['mix.pw','stem'],720,8,8,activation='none')
    add('finish.dw','dw3x3',['mix_add'],720,8,8)
    add('rgb','pw1x1',['finish.dw'],720,8,3,activation='clamp_rgb')
    return dict(nodes=nodes,macs=sum(n['macs'] for n in nodes),production=False)


class RegionNet(nn.Module):
    def __init__(self):
        super().__init__();self.nodes=plan()['nodes'];self.layers=nn.ModuleDict()
        for n in self.nodes:
            if not n['kernel']:continue
            parts=[nn.ReplicationPad2d(1)] if n['kernel']==3 else []
            conv=nn.Conv2d(n['input_channels'],n['channels'],n['kernel'],n['stride'],groups=n['groups'],bias=True)
            nn.init.kaiming_normal_(conv.weight,nonlinearity='relu');nn.init.zeros_(conv.bias);parts.append(conv)
            if n['activation']=='relu':parts.append(nn.ReLU())
            self.layers[n['id'].replace('.','__')]=nn.Sequential(*parts)
        # Exact input colors at initialization, carried by learned native
        # features, not an external source-image blend after the generator.
        with torch.no_grad():
            stem=self.conv('stem');stem.weight[:3].zero_();stem.bias[:3]=1
            for c in range(3):stem.weight[c,c,1,1]=1
            mix=self.conv('mix.pw');mix.weight.zero_();mix.bias.zero_()
            finish=self.conv('finish.dw');finish.weight.zero_();finish.weight[:,0,1,1]=1;finish.bias.zero_()
            head=self.conv('rgb');head.weight.zero_();head.bias.fill_(-1)
            for c in range(3):head.weight[c,c,0,0]=1
        self.eval()

    def conv(self,name):return next(m for m in self.layers[name.replace('.','__')] if isinstance(m,nn.Conv2d))

    def forward(self,x):
        values={'input':x}
        for n in self.nodes:
            a=values[n['inputs'][0]]
            if n['kind']=='residual_add':v=a+values[n['inputs'][1]]
            elif n['kind']=='nearest_x2':v=a.repeat_interleave(2,-1).repeat_interleave(2,-2)
            else:
                v=self.layers[n['id'].replace('.','__')](a)
                if n['activation']=='clamp_rgb':v=v.clamp(-1,127/128)
            values[n['id']]=v
        return values['rgb']


def fused_plan():
    out=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            pw=out.pop();assert pw['id']==n['inputs'][0]
            n.update(kind='pw_fused',parameter_node=pw['id'],inputs=[pw['inputs'][0],n['inputs'][1]],
                input_shape_hwc=pw['input_shape_hwc'],input_channels=pw['input_channels'],terms=pw['terms'],macs=pw['macs'])
        out.append(n)
    # The reusable diagnostic tile builder expects the final RGB node's name.
    out[-1]['id']='rgb_head'
    return dict(nodes=out)
