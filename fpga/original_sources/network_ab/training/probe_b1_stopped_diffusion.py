"""Three edge-stopping flux updates encoded in the existing B1 residuals.

The triangular flux is a custom piecewise-linear approximation, not the
original Perona/Malik conductance. Its four clipped ramps use existing ReLUs
and INT8 saturation. No external diffusion, palette lookup or filter is used
when the exported bytes are executed. There is no minimum-area guarantee.
"""
import argparse
import shutil
import torch
from torch.nn import functional as F
from .seed_b1_binary_regions import build as base
from . import train_b1_minimum_area as m
from .runtime import configure_workspace, write_json, file_hash
from .engine import set_reproducible
from .publish_b1_style_recovery import export


@torch.no_grad()
def build(threshold=16., unit=6.):
    net=base()
    w,b=net.weight['down2__pw'],net.bias['down2__pw'];w.zero_();b.zero_()
    for axis in range(2):
        for c in range(3):
            for ramp in range(4):w[axis*12+c*4+ramp,c,0,0]=1
    for stage in range(1,4):
        for half in (1,2):
            for kind in ('dw','pw'):
                key=f'res{stage}_conv{half}__{kind}'
                net.weight[key].zero_();net.bias[key].zero_()
        w,b=net.weight[f'res{stage}_conv1__dw'],net.bias[f'res{stage}_conv1__dw']
        for axis in range(2):
            for j in range(axis*12,(axis+1)*12):
                w[j,0,1,1]=-.25
                w[j,0,1 if axis==0 else 2,2 if axis==0 else 1]=.25;b[j]=32
        w,b=net.weight[f'res{stage}_conv1__pw'],net.bias[f'res{stage}_conv1__pw']
        for j in range(24):
            ramp=j%4;sign=1 if ramp<2 else -1
            w[j,j,0,0]=sign*4*127/threshold
            b[j]=-sign*128*127/threshold-(127 if ramp%2 else 0)
        w,b=net.weight[f'res{stage}_conv2__dw'],net.bias[f'res{stage}_conv2__dw']
        for axis in range(2):
            for j in range(axis*12,(axis+1)*12):
                w[j,0,1,1]=.5
                w[j,0,1 if axis==0 else 0,0 if axis==0 else 1]=-.5;b[j]=64
        w,b=net.weight[f'res{stage}_conv2__pw'],net.bias[f'res{stage}_conv2__pw']
        for output in range(24):
            c=(output%12)//4
            for axis in range(2):
                for ramp,sign in enumerate((1,-1,-1,1)):
                    w[output,axis*12+c*4+ramp,0,0]=sign*.25*threshold*2/127
    for key in ('up1_low_dw','up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    for j in range(24):net.weight['up1_low_dw'][j,0,1,1]=1
    for c in range(3):net.weight['up1_low_pw'][c,c*4,0,0]=1
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(3):
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=1
    units=kernel.new_tensor((unit,3.,3.));phase=kernel.new_tensor((0.,(-64)%3,(-64)%3))
    for c in range(3):
        net.weight['up2_low_pw'][c,c,0,0]=1/units[c]
        net.bias['up2_low_pw'][c]=phase[c]/units[c]
        net.weight['up2_high_dw'][c,0]=kernel
    # Sparse, thresholded large boundaries, drawn from the same cleaned Y.
    for c in range(3,7):net.weight['up2_low_pw'][c,0,0,0]=1
    sobel=kernel.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/4
    for c,k in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):
        net.weight['up2_high_dw'][c,0]=k;net.bias['up2_high_dw'][c]=-3
    matrix=kernel.new_tensor([[1.04,1.08,-1.08],[1.04,0,1.08],[1.04,-1.08,-1.08]])
    net.weight['rgb_head'][:,:3,0,0]=2*matrix*units[None]
    net.weight['rgb_head'][:,3:7,0,0]=-.4
    net.bias['rgb_head'].copy_(-2*(matrix*(64+phase)[None]).sum(1)+kernel.new_tensor((-2.,0.,3.)))
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.no_grad()
def levelset_decoder(net, smooth_coverage=False):
    """Interpolate region coverage first; sharpen it before mixing colours."""
    from .seed_b1_clear_regions import BASES, INCREMENTS
    for key in ('up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(12):
        net.weight['up1_low_pw'][c,0,0,0]=16
        net.bias['up1_low_pw'][c]=63.5-16*(10+10*c)
        net.weight['up1_high_dw'][c,0]=kernel
        if smooth_coverage:net.weight['up2_low_dw'][c,0]=6*kernel
        else:net.weight['up2_low_dw'][c,0,1,1]=6
        net.bias['up2_low_dw'][c]=-5*63.5
    for c,j in ((12,4),(13,8)):
        net.weight['up1_low_pw'][c,j,0,0]=1
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=1
    for c,(lo,inc) in enumerate(zip(BASES,INCREMENTS)):
        net.weight['up2_low_pw'][c,:12,0,0]=10/inc
        net.bias['up2_low_pw'][c]=(5-lo)*127/inc
        net.weight['up2_high_dw'][c,0]=kernel*3/127
        net.weight['rgb_head'][:,c,0,0]=2*inc/3
    for c in (6,7):
        net.weight['up2_low_pw'][c,c+6,0,0]=1
        net.weight['up2_high_dw'][c,0]=kernel*.25
    net.weight['rgb_head'][:,6,0,0]=kernel.new_tensor((8,0,-8))
    net.weight['rgb_head'][:,7,0,0]=kernel.new_tensor((-8,8,-8))
    net.bias['rgb_head'].copy_(kernel.new_tensor((-112,-240,144)))
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.no_grad()
def coverage_colour_decoder(net):
    """Keep14 binary Y regions until360p; mix once, then boundary AA only.

Re-quantizing a reconstructed ramp creates concentric intermediate bands.
This variant does not quantize the mixed colour again in the decoder.
"""
    for key in ('up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(14):
        net.weight['up1_low_pw'][c,0,0,0]=16
        net.bias['up1_low_pw'][c]=63.5-16*(8+8*c)
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=6
        net.bias['up2_low_dw'][c]=-5*63.5
    for c,j in ((14,4),(15,8)):
        net.weight['up1_low_pw'][c,j,0,0]=1
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=1
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw']
    w[:3,:14,0,0]=8/127
    w[0,14,0,0]=1;w[0,15,0,0]=-1;b[0]=4
    w[1,15,0,0]=1;b[1]=4-64
    w[2,14:,0,0]=-1;b[2]=4+128
    for c in range(3):
        net.weight['up2_high_dw'][c,0]=kernel
        net.weight['rgb_head'][c,c,0,0]=2
    net.bias['rgb_head'].fill_(-128)
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.no_grad()
def bitplane_decoder(net, aa_states=2):
    """Transport15 ordered brightness regions through five binary channels.

The most significant bit uses two channels so the existing signed INT8
coefficient range need not represent a coefficient of128. Remaining channels
carry chroma. Threshold/coverage rounding happens at720p before RGB mixing.
"""
    assert aa_states in (1,2,4)
    coverage_colour_decoder(net)
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c,bit in enumerate((0,1,2,3,3)):
        for threshold in range(1,15):
            diff=((threshold>>bit)&1)-(((threshold-1)>>bit)&1)
            net.weight['up2_low_pw'][c,threshold-1,0,0]=diff
        net.weight['up2_high_dw'][c,0]=kernel*aa_states/127
        net.weight['rgb_head'][:,c,0,0]=(16*(2**bit)/(2 if bit==3 else 1))/aa_states
    for c,j in ((5,14),(6,15)):
        net.weight['up2_low_pw'][c,j,0,0]=1
        net.weight['up2_high_dw'][c,0]=kernel
    net.weight['rgb_head'][:,5,0,0]=kernel.new_tensor((2,0,-2))
    net.weight['rgb_head'][:,6,0,0]=kernel.new_tensor((-2,2,-2))
    net.bias['rgb_head'].copy_(kernel.new_tensor((-120,-248,136)))
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    # Exact algebraic check of the code, independent of image appearance.
    codes=torch.arange(15,device=kernel.device)
    unary=(codes[:,None]>=torch.arange(1,15,device=kernel.device)).float()
    weights=net.weight['up2_low_pw'][:5,:14,0,0]
    bits=unary@weights.T
    reconstructed=bits@kernel.new_tensor((1,2,4,4,4))
    assert torch.equal(reconstructed,codes.float())
    return net


@torch.no_grad()
def paired_coverage_decoder(net):
    """Monotone adjacent pairs avoid bit-carry speckles at mixed boundaries."""
    levelset_decoder(net)
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(6):
        net.weight['up2_low_pw'][c,2*c:2*c+2,0,0]=.5
        net.weight['up2_high_dw'][c,0]=kernel*4/127
        net.weight['rgb_head'][:,c,0,0]=10
    for c in (6,7):
        net.weight['up2_low_pw'][c,c+6,0,0]=1
        net.weight['up2_high_dw'][c,0]=kernel
    net.weight['rgb_head'][:,6,0,0]=kernel.new_tensor((2,0,-2))
    net.weight['rgb_head'][:,7,0,0]=kernel.new_tensor((-2,2,-2))
    net.bias['rgb_head'].copy_(kernel.new_tensor((-118,-246,138)))
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.no_grad()
def colour_region_decoder(net):
    """Use matching coverage geometry for both brightness and chroma.

Continuous, blurred chroma crossing a sharp luminance boundary creates coloured
rings. This variant transports8Y+4Co+4Cg monotone masks to720p in8 adjacent pairs.
"""
    for key in ('up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    levels=[(0,q) for q in (14,28,42,56,70,84,98,112)]
    levels += [(4,q) for q in (49,59,69,79)]+[(8,q) for q in (49,59,69,79)]
    for c,(j,q) in enumerate(levels):
        net.weight['up1_low_pw'][c,j,0,0]=16;net.bias['up1_low_pw'][c]=63.5-16*q
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=6;net.bias['up2_low_dw'][c]=-5*63.5
    for c in range(8):
        net.weight['up2_low_pw'][c,2*c:2*c+2,0,0]=.5
        net.weight['up2_high_dw'][c,0]=kernel*4/127
    net.weight['rgb_head'][:,:4,0,0]=14
    net.weight['rgb_head'][:,4:6,0,0]=kernel.new_tensor((10,0,-10))[:,None]
    net.weight['rgb_head'][:,6:8,0,0]=kernel.new_tensor((-10,10,-10))[:,None]
    net.bias['rgb_head'].copy_(kernel.new_tensor((-114,-154,-34)))
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.no_grad()
def opened_colour_regions(net, size=3):
    """Opening/closing per monotone mask, using the last residual and low DW.

An interior component of each opened binary plane contains at least size²
180p cells. This is NOT a minimum RGB-intersection area guarantee. Thin objects
can also be removed, which is a material visual limitation of this probe.
"""
    assert size in (2,3)
    colour_region_decoder(net)
    levels=[(0,q) for q in (14,28,42,56,70,84,98,112)]
    levels += [(4,q) for q in (49,59,69,79)]+[(8,q) for q in (49,59,69,79)]
    for key in ('res3_conv1__dw','res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw','up1_low_pw'):
        net.weight[key].zero_();net.bias[key].zero_()
    for c in range(24):net.weight['res3_conv1__dw'][c,0,1,1]=1
    for c,(j,q) in enumerate(levels):
        net.weight['res3_conv1__pw'][c,j,0,0]=127
        net.bias['res3_conv1__pw'][c]=-127*(q-1)
        closing=(c>=8 and q<64)
        net.weight['res3_conv2__dw'][c,0,:size,:size]=1
        net.bias['res3_conv2__dw'][c]=0 if closing else -127*(size*size-1)
        net.weight['res3_conv2__pw'][c,c,0,0]=1
        net.weight['up1_low_dw'][c,0,3-size:,3-size:]=1
        net.bias['up1_low_dw'][c]=-127*(size*size-1) if closing else 0
        net.weight['up1_low_pw'][c,c,0,0]=1
    net.package['residual_encoding']['res3_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    net.region_opening=size
    return net


@torch.no_grad()
def full_gradient_range(net):
    """Use the full positive activation range for signed local differences.

0.25*d+32 wasted half the available range;0.5*d+64 halves the rounding
interval. The following affine map is compensated, preserving the FP32 rule.
"""
    for stage in range(1,4):
        kd=f'res{stage}_conv1__dw';kp=f'res{stage}_conv1__pw'
        net.weight[kd][:24]*=2;net.bias[kd][:24]=64
        net.weight[kp]/=2
        for key in (kd,kp):
            w=net.weight[key]
            scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
            getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


def reference_update(u,threshold):
    update=torch.zeros_like(u)
    for axis in (-1,-2):
        pad=(0,1,0,0) if axis==-1 else (0,0,0,1)
        padded=F.pad(u,pad,mode='replicate')
        neighbour=padded[...,1:] if axis==-1 else padded[...,1:,:]
        delta=neighbour-u
        flux=delta.sign()*torch.minimum(delta.abs(),2*threshold-delta.abs()).clamp_min(0)
        pad=(1,0,0,0) if axis==-1 else (0,0,1,0)
        padded=F.pad(flux,pad,mode='replicate')
        previous=padded[...,:-1] if axis==-1 else padded[...,:-1,:]
        update+=(flux-previous)*.25
    return u+update


@torch.inference_mode()
def unit_check(net,threshold):
    net.quantized=False
    x=torch.randn(1,3,96,96,device='cuda')*.15
    nodes=net(x,all_nodes=True);u=nodes['down2.pw'][:,[0,4,8]]
    errors=[]
    for stage in range(1,3 if getattr(net,'region_opening',0) else 4):
        wanted=reference_update(u,threshold)
        got=nodes[f'res{stage}_add'][:,[0,4,8]]
        errors.append(float((wanted-got).abs().max()));u=got
    assert max(errors)<.003,errors
    constant=torch.full((1,3,24,32),40.,device='cuda')
    step=constant.clone();step[...,16:]=120
    assert torch.equal(reference_update(constant,threshold),constant)
    assert torch.equal(reference_update(step,threshold),step)
    net.quantized=True
    return dict(fp32_update_errors=errors,constant_exact=True,strong_step_exact=True,
                border='Replicated flux at the existing hardware boundary',not_minimum_area_proof=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--threshold',type=float,default=16.);ap.add_argument('--unit',type=float,default=6.)
    ap.add_argument('--levelset',action='store_true')
    ap.add_argument('--smooth-coverage',action='store_true')
    ap.add_argument('--coverage-colour',action='store_true')
    ap.add_argument('--bitplanes',type=int,choices=(1,2,4),default=0)
    ap.add_argument('--paired-coverage',action='store_true')
    ap.add_argument('--colour-regions',action='store_true')
    ap.add_argument('--opening',type=int,choices=(2,3),default=0)
    ap.add_argument('--full-gradient-range',action='store_true')
    a=ap.parse_args();configure_workspace();set_reproducible(20261006117)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);net=build(a.threshold,a.unit)
    if a.full_gradient_range:full_gradient_range(net)
    if a.levelset:levelset_decoder(net,a.smooth_coverage)
    if a.coverage_colour:coverage_colour_decoder(net)
    if a.bitplanes:bitplane_decoder(net,a.bitplanes)
    if a.paired_coverage:paired_coverage_decoder(net)
    if a.colour_regions:colour_region_decoder(net)
    if a.opening:opened_colour_regions(net,a.opening)
    check=unit_check(net,a.threshold);rows=m.load_data()['validation']
    export(net,rows,dest/'package');m.t.preview(net,rows,dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=m.t.r.CONTRACT_SHA,
                    residual_encoding=net.package['residual_encoding'],field_node='res2_add' if a.opening else 'res3_add',
                    parameter_initialization_only=True),dest/'initial.pt')
    shutil.copy2(__file__,dest/'probe_source.py')
    write_json(dest/'review.json',dict(args=vars(a),unit_check=check,code_sha256=file_hash(__file__),
        added_operators=0,minimum_area_guaranteed=False,visual_accepted=False,
        reference='https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf',
        custom_triangular_flux=True,training_steps=0))
    print('STOPPED_DIFFUSION_PROBE',dest,check,flush=True)


if __name__=='__main__':main()
