"""Measure whether B1 actually uses its final720 clipping nonlinearity."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def preactivation(net,key,x):
    multiplier=getattr(net,key+'_m');scale=(multiplier/131072).float().flatten()
    weight=net.round(net.weight[key]/scale[:,None,None,None]).clamp(-127,127)
    bias=net.round(net.bias[key]/scale).double()[None,:,None,None]
    groups=x.shape[1] if weight.shape[1]==1 and weight.shape[-1]==3 else 1
    if weight.shape[-1]==3:x=F.pad(x,(1,1,1,1),mode='replicate')
    value=F.conv2d(x,weight,groups=groups)
    return net.round((value.double()+bias).clamp(-(2**31),2**31-1)*multiplier/131072).float()


def nullspace_regression():
    generator=torch.Generator().manual_seed(107)
    low=torch.randn(1,1,16,20,generator=generator,dtype=torch.float64)
    high=low.repeat_interleave(2,-2).repeat_interleave(2,-1)
    kernel=torch.randn(1,1,3,3,generator=generator,dtype=torch.float64)
    z=F.conv2d(F.pad(high,(1,1,1,1),mode='circular'),kernel)
    signs=(-1.)**torch.arange(z.shape[-1],dtype=z.dtype)
    linear=float((z*signs).sum().abs());nonlinear=float((z.clamp(0,1)*signs).sum().abs())
    assert linear<1e-10 and nonlinear>1e-4
    return dict(unrounded_periodic_linear_alternating_sum=linear,clipped_sum=nonlinear,
                limitation='linear idealisation only; rounding and clipping change the exact property')


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);test=nullspace_regression();_,net=load(a.checkpoint);records=[]
    for row in m.load_data(a.data)['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True);pre=preactivation(net,'up2_high_dw',nodes['up2_nn'])
        assert torch.equal(pre.clamp(0,127),nodes['up2_high_dw'])
        original=preactivation(net,'rgb_head',nodes['up2_high_dw']).clamp(-128,127)
        assert torch.equal(original,nodes['rgb_head'])
        no_clip=preactivation(net,'rgb_head',pre).clamp(-128,127)
        records.append(dict(id=row['id'],below0_fraction=(pre<0).float().mean((0,2,3)).tolist(),
            above127_fraction=(pre>127).float().mean((0,2,3)).tolist(),
            bypass_last_relu_clip_rgb_mae=float((no_clip-original).abs().mean()),
            bypass_changed_fraction=float((no_clip!=original).any(1).float().mean())))
    write_json(a.out,dict(records=records,linear_upsampling_unit_test=test,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),no_parameter_update=True,bypass_is_diagnostic_only=True))
    print('LAST720_NONLINEARITY',records,flush=True)


if __name__=='__main__':main()
