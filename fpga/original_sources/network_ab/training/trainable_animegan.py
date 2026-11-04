"""Differentiable import of the pinned author graph; preserves ONNX-9 semantics."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .runtime import file_hash
from .animegan_teacher import REVISION, MODEL_SHA


class GeneralPad(nn.Module):
    def __init__(self,pads,mode='constant',value=0):
        super().__init__();self.pads=list(pads);self.mode=mode;self.value=value

    def forward(self,x):
        rank=x.ndim;before=self.pads[:rank];after=self.pads[rank:]
        if self.mode=='constant':
            pads=[v for pair in reversed(list(zip(before,after))) for v in pair]
            return F.pad(x,pads,value=self.value)
        for axis,(a,b) in enumerate(zip(before,after)):
            if not (a or b):continue
            length=x.shape[axis]
            if self.mode=='reflect':
                if min(a,b)<0 or max(a,b)>=length:raise ValueError('Unsupported reflect padding')
                index=torch.cat([torch.arange(a,0,-1,device=x.device),torch.arange(length,device=x.device),
                                 torch.arange(length-2,length-2-b,-1,device=x.device)])
            elif self.mode=='edge':index=torch.arange(-a,length+b,device=x.device).clamp(0,length-1)
            else:raise ValueError(self.mode)
            x=x.index_select(axis,index.long())
        return x


class Upsample9(nn.Module):
    def __init__(self,mode):super().__init__();self.mode=mode
    def forward(self,x,scales):
        scale=scales.detach().cpu().tolist()
        if len(scale)!=4 or scale[:2]!=[1.,1.]:raise ValueError('Expected NCHW spatial upsample: '+str(scale))
        h,w=x.shape[-2:];oh,ow=int(h*scale[2]),int(w*scale[3])
        if self.mode=='nearest':return F.interpolate(x,size=(oh,ow),mode='nearest')
        if self.mode!='linear':raise ValueError(self.mode)
        # ONNX Upsample-9 uses asymmetric coordinates (TF legacy resize), not half-pixel.
        yy=torch.arange(oh,device=x.device,dtype=x.dtype)/scale[2]
        xx=torch.arange(ow,device=x.device,dtype=x.dtype)/scale[3]
        gy,gx=torch.meshgrid(2*yy/max(h-1,1)-1,2*xx/max(w-1,1)-1,indexing='ij')
        grid=torch.stack([gx,gy],-1)[None].expand(x.shape[0],-1,-1,-1)
        return F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=True)


class Slice1(nn.Module):
    def __init__(self,starts,ends,axes=None):
        super().__init__();self.starts=starts;self.ends=ends;self.axes=axes or list(range(len(starts)))
    def forward(self,x):
        slices=[slice(None)]*x.ndim
        for axis,a,b in zip(self.axes,self.starts,self.ends):slices[axis]=slice(a,b)
        return x[tuple(slices)]


def author_graph(root):
    import onnx
    from onnx2torch import convert
    from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
    from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
    from onnx2torch.node_converters.global_average_pool import OnnxGlobalAveragePool
    def pad(node,graph):
        a=node.attributes
        return OperationConverterResult(GeneralPad(a['pads'],a.get('mode','constant'),a.get('value',0)),onnx_mapping_from_node(node=node))
    def upsample(node,graph):
        return OperationConverterResult(Upsample9(node.attributes.get('mode','nearest')),onnx_mapping_from_node(node=node))
    def slicer(node,graph):
        a=node.attributes
        return OperationConverterResult(Slice1(a['starts'],a['ends'],a.get('axes')),onnx_mapping_from_node(node=node))
    def global_pool(node,graph):
        # Legacy value_info contains unknown ranks; derive spatial axes at runtime.
        return OperationConverterResult(OnnxGlobalAveragePool(),onnx_mapping_from_node(node=node))
    _CONVERTER_REGISTRY[OperationDescription('','Pad',2)]=pad
    _CONVERTER_REGISTRY[OperationDescription('','Upsample',9)]=upsample
    _CONVERTER_REGISTRY[OperationDescription('','Slice',1)]=slicer
    _CONVERTER_REGISTRY[OperationDescription('','GlobalAveragePool',1)]=global_pool
    path=Path(root)/'cache/animegan-probe'/REVISION/'AnimeGANv3_Hayao_36.onnx'
    if file_hash(path)!=MODEL_SHA:raise ValueError('Author model drift')
    return convert(onnx.load(path))


class TrainableAnimeGAN(nn.Module):
    def __init__(self,root):super().__init__();self.generator=author_graph(root)
    def forward(self,x):
        # External interface is existing project NCHW /128-1; author uses NHWC /127.5-1.
        z=((x+1)*(128/127.5)-1).permute(0,2,3,1)
        y=self.generator(z)
        return ((y+1)*(127.5/128)-1).permute(0,3,1,2)
