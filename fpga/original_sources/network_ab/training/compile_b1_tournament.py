"""Exact16-way argmax on full0..127 scores in the existing six residual PWs.

Store binary tournament choices instead of retaining16 original scores. Each
round carries max(a,b)=ReLU(a-b)+b, and previous choices. The last PW combines
four path choices into a one-hot leaf. Ties choose the earlier colour index.
"""
import torch
from torch.nn import functional as F
from .probe_b1_decoder_coverage import fragment
from .repair_b1_region_partition import repair


@torch.no_grad()
def compile_tournament(net):
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['res1_conv2__dw'].zero_();net.weight['res1_conv2__dw'][:16,0]=kernel;net.bias['res1_conv2__dw'].zero_()
    stages=('res1_conv2__pw','res2_conv1__pw','res2_conv2__pw','res3_conv1__pw')
    previous=None;old_choices=0
    for stage,key in enumerate(stages):
        net.weight[key].zero_();net.bias[key].zero_();n=8>>stage
        for j in range(n):
            if stage==0:left=[2*j];right=[2*j+1]
            else:
                p=n*2;left=[p+2*j,2*p+2*j];right=[p+2*j+1,2*p+2*j+1]
            for src in left:net.weight[key][j,src,0,0]=127;net.weight[key][n+j,src,0,0]=1
            for src in right:
                net.weight[key][j,src,0,0]=-127;net.weight[key][n+j,src,0,0]=-1;net.weight[key][2*n+j,src,0,0]=1
            net.bias[key][j]=127
        if stage:
            p=2*n;srcs=list(range(p))+list(range(3*p,3*p+old_choices))
            for j,src in enumerate(srcs):net.weight[key][3*n+j,src,0,0]=1
            old_choices+=p
    key='res3_conv2__pw';net.weight[key].zero_();net.bias[key].zero_()
    for leaf in range(16):
        mask_indices=(9+leaf//2,5+leaf//4,3+leaf//8,0);positive=0
        for level,src in enumerate(mask_indices):
            sign=1 if ((leaf>>level)&1)==0 else -1
            net.weight[key][leaf,src,0,0]=sign;positive+=sign==1
        net.bias[key][leaf]=127*(1-positive)
    for key in ('res1_conv2__dw',)+stages+('res3_conv2__pw',):
        w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
    repair(net)


@torch.no_grad()
def verify(net):
    gen=torch.Generator(device='cuda').manual_seed(8106)
    z=torch.zeros(256,32,8,8,device='cuda');z[:,:16]=torch.randint(0,128,(256,16,8,8),generator=gen,device='cuda')
    z[0,:16]=64;z[1,:16]=0;z[2,:16]=127
    result=fragment(net,z,'res1_conv2.dw')['up1_low_pw']
    expected=F.one_hot(z[:,:16].argmax(1),16).permute(0,3,1,2)*127
    error=float((result-expected).abs().max());assert error==0,error
    return dict(full_precision_score_range=[0,127],random_and_tie_cases=256*8*8,error=error,ties='earliest_colour')
