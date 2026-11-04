"""Physically prune dependency groups of the recovered half-width teacher.

Different blocks can keep different channels. Only residual-add dependencies
and each LADE main/style pair are tied. The original model stays immutable.
"""
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from .prune_animation_teacher import build_pruned
from .runtime import file_hash

RUN='importance-pruning-20261003'
PARENT='teacher-pruning-calibrated-20261003/keep50_recovery/pruned_teacher_fp32.pt'
PARENT_SHA='d58f6103cdc31affac8c9309a718d150c278fe9573b839f59d07e578859547f3'
WIDTHS=dict(b0=16,b1=16,b2=32,b3=32,b4=64,b5=64,b6=64,attn_projection=64,attn_memory=64,d1=64,d3=32,d5=16)


def conv_name(branch,index):
    suffix='' if index==0 else '_'+str(index)
    return 'generator.generator_1/'+branch+'/Conv'+suffix+'/Conv2D'


def layer_groups():
    result={};previous=None
    for i in range(7):
        group='b'+str(i)
        result[conv_name('base',2*i)]=(previous,group)
        result[conv_name('base',2*i+1)]=(group,group)
        previous=group
    prefix='generator.generator_1/main/External_attention/'
    result[prefix+'Conv/Conv2D']=('b6','attn_projection')
    result[prefix+'conv1d']=('attn_projection','attn_memory')
    result[prefix+'conv1d_1']=('attn_memory','attn_projection')
    result[prefix+'batchnorm/mul_1']=('attn_projection','b6')
    previous='b6'
    for i,group in enumerate(['b4','d1','b2','d3','b0','d5']):
        result[conv_name('main',2*i)]=(previous,group)
        result[conv_name('main',2*i+1)]=(group,group)
        previous=group
    result[conv_name('main',12)]=('d5',None)
    return result


class ChannelBroadcast(nn.Module):
    def forward(self,x,unused_shape):
        assert x.ndim==4 and x.shape[2:]==(1,1)
        return x.reshape(x.shape[0],1,1,x.shape[1])


class AttentionFlatten(nn.Module):
    def forward(self,x,unused_shape):
        # Input is NHWC; projection width need not equal encoder width.
        return x.reshape(x.shape[0],-1,x.shape[-1])


class AttentionUnflatten(nn.Module):
    def forward(self,x,shape):
        size=shape.detach().cpu().tolist()
        size[0]=x.shape[0];size[-1]=x.shape[-1]
        return x.reshape(size)


@lru_cache(maxsize=1)
def parent(root):
    path=Path(root)/'runs'/PARENT;assert file_hash(path)==PARENT_SHA
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=build_pruned(root,state['indices']);net.load_state_dict(state['generator'])
    return net.eval()


def full_plan():return {k:list(range(c)) for k,c in WIDTHS.items()}


def transfer(net,base,plan,compensation=None):
    for name,(ig,og) in layer_groups().items():
        old=base.get_submodule(name);new=net.get_submodule(name)
        ii=plan[ig] if ig else list(range(old.in_channels))
        oo=plan[og] if og else list(range(old.out_channels))
        w=old.weight.detach().cpu()[oo]
        if compensation is not None and len(ii)<old.in_channels:
            xx=compensation[name].double();kk=xx[ii][:,ii]
            ridge=float(kk.trace()/len(ii))*1e-4+1e-12
            A=torch.linalg.solve(kk+ridge*torch.eye(len(ii),dtype=torch.float64),xx[ii]).T
            A[ii]=torch.eye(len(ii),dtype=torch.float64)
            w=torch.einsum('oihw,ik->okhw',w.double(),A).float()
        else:w=w[:,ii]
        assert new.weight.shape==w.shape
        with torch.no_grad():
            new.weight.copy_(w)
            if new.bias is not None:new.bias.copy_(old.bias.detach().cpu()[oo])


def build_group_pruned(root,plan):
    assert set(plan)==set(WIDTHS)
    for name,c in WIDTHS.items():
        assert plan[name] and plan[name]==sorted(set(plan[name])) and 0<=min(plan[name])<=max(plan[name])<c
    base=parent(root)
    # Rebuild rather than mutating the cached parent or sharing parameters.
    state=torch.load(Path(root)/'runs'/PARENT,map_location='cpu',weights_only=True)
    net=build_pruned(root,state['indices']);net.load_state_dict(state['generator'])
    mapping=layer_groups();actual={n for n,m in net.named_modules() if isinstance(m,nn.Conv2d)}
    assert actual==set(mapping)
    for name,(ig,og) in mapping.items():
        m=base.get_submodule(name)
        ci=len(plan[ig]) if ig else m.in_channels;co=len(plan[og]) if og else m.out_channels
        owner,leaf=name.rsplit('.',1)
        net.get_submodule(owner)._modules[leaf]=nn.Conv2d(ci,co,m.kernel_size,m.stride,m.padding,m.dilation,bias=m.bias is not None)
    transfer(net,base,plan)
    replaced=0
    for node in net.generator.graph.nodes:
        if node.op!='call_module' or type(net.generator.get_submodule(node.target)).__name__!='OnnxReshape':continue
        shape_node=node.args[1]
        if getattr(shape_node,'op',None)!='get_attr':continue
        value=net.generator
        for part in shape_node.target.split('.'):value=getattr(value,part)
        if value.numel()==4 and value.tolist()[:3] in ([1,1,1],[-1,1,1]):
            net.generator._modules[node.target]=ChannelBroadcast();replaced+=1
    assert replaced==26,replaced
    net.generator._modules['generator_1/main/External_attention/Reshape']=AttentionFlatten()
    net.generator._modules['generator_1/main/External_attention/Reshape_1']=AttentionUnflatten()
    return net.eval()


def cost(plan,spatial):
    base=parent(Path('E:/fpga/model-training-workspace'))
    total=0;parameters=0
    for name,(ig,og) in layer_groups().items():
        m=base.get_submodule(name);ci=len(plan[ig]) if ig else m.in_channels;co=len(plan[og]) if og else m.out_channels
        weights=ci*co*m.kernel_size[0]*m.kernel_size[1]
        total+=spatial[name]*weights;parameters+=weights+(co if m.bias is not None else 0)
    return dict(conv_macs=total,parameters=parameters)
