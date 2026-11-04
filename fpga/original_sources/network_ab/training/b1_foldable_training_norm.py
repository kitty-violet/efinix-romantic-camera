"""Training-only BatchNorm coordinates, folded into existing B1 convolutions.

The reference warm function, internal signed-byte clipping, residual gains,
and spatial graph are preserved in eval FP32. Batch statistics never run in
deployment. Quantization is a separately measured step after folding.
Reference: https://docs.pytorch.org/docs/main/generated/torch.nn.utils.fuse_conv_bn_eval.html
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class FoldableTrainingNorm(nn.Module):
    unit=32.

    def __init__(self,warm,calibration):
        super().__init__()
        self.template=copy.deepcopy(warm).cpu().requires_grad_(False)
        self.plan=copy.deepcopy(warm.plan)
        self.residual=copy.deepcopy(warm.package['residual_encoding'])
        self.conv=nn.ModuleDict();self.norm=nn.ModuleDict()
        device=calibration.device
        # Calibrate pre-affine responses in the exact warm FP32 function;
        # matching train-batch statistics keeps this diagnostic's starting
        # point controlled rather than silently replacing its function.
        base=copy.deepcopy(warm).to(device);base.quantized=False
        with torch.no_grad():nodes=base(calibration,all_nodes=True)
        nodes['input']=(calibration*128).clamp(-128,127)
        for row in self.plan['nodes']:
            k=row['id'].replace('.','__')
            if k not in warm.weight:continue
            final=row['id']=='rgb_head'
            conv=nn.Conv2d(row['input_channels'],row['channels'],row['kernel'],stride=row['stride'],groups=row['groups'],bias=final).to(device)
            with torch.no_grad():
                conv.weight.copy_(base.weight[k])
                if final:conv.bias.copy_(base.bias[k]/self.unit)
            self.conv[k]=conv
            if not final:
                bn=nn.BatchNorm2d(row['channels'],eps=1e-5,momentum=.05).to(device)
                with torch.no_grad():
                    x=nodes[row['inputs'][0]]/self.unit
                    if row['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
                    z=conv(x);mu=z.mean((0,2,3));var=z.var((0,2,3),unbiased=False)
                    bn.running_mean.copy_(mu);bn.running_var.copy_(var)
                    bn.weight.copy_((var+bn.eps).sqrt());bn.bias.copy_(mu+base.bias[k]/self.unit)
                self.norm[k]=bn
        # template is serialization/config only, never part of optimization.
        self.template.eval()

    def train(self,mode=True):
        super().train(mode);self.template.eval();return self

    def forward(self,x,all_nodes=False):
        nodes={'input':(x*128).clamp(-128,127)}
        for row in self.plan['nodes']:
            name=row['id'];k=name.replace('.','__');a=nodes[row['inputs'][0]]
            if k in self.conv:
                z=a/self.unit
                if row['kernel']==3:z=F.pad(z,(1,1,1,1),mode='replicate')
                z=self.conv[k](z)
                if k in self.norm:z=self.norm[k](z)
                y=(z*self.unit).clamp(-128,127)
                if row['activation']=='relu':y=y.clamp_min(0)
            elif row['kind']=='nearest_x2':
                y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                z=nodes[row['inputs'][1]];q=self.residual[name]
                small,other=(a,z) if q['main_scaled'] else (z,a)
                y=(((small*q['multiplier']/131072).clamp(-128,127)+other)*q['post_multiplier']/131072).clamp(-128,127)
            nodes[name]=y
        return nodes if all_nodes else nodes['rgb_head']/128

    @torch.no_grad()
    def fold(self):
        assert not self.training,'Freeze running statistics before folding'
        net=copy.deepcopy(self.template).to(next(self.conv.parameters()).device)
        for k,conv in self.conv.items():
            if k in self.norm:
                bn=self.norm[k];gain=bn.weight/(bn.running_var+bn.eps).sqrt()
                w=conv.weight*gain[:,None,None,None]
                b=self.unit*(bn.bias-bn.running_mean*gain)
            else:w=conv.weight;b=conv.bias*self.unit
            net.weight[k].copy_(w);net.bias[k].copy_(b)
        net.quantized=False;return net

    @torch.no_grad()
    def calibrate_running(self,x):
        """Exact local diagnostic stats in topological order, no test data."""
        self.eval();hooks=[]
        for bn in self.norm.values():
            def update(module,inputs):
                z=inputs[0]
                module.running_mean.copy_(z.mean((0,2,3)))
                module.running_var.copy_(z.var((0,2,3),unbiased=False))
            hooks.append(bn.register_forward_pre_hook(update))
        try:self(x)
        finally:
            for hook in hooks:hook.remove()
