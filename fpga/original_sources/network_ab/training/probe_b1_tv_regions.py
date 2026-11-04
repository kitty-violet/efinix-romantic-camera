"""Three projected dual-gradient stages mapped into existing B1 convolutions.

Anisotropic box projection, INT8 rounding and three iterations: this is inspired
by total-variation optimization, not a converged isotropic Chambolle solver.
No Gaussian convolution is used inside the three region update stages.
"""
import argparse
import shutil
import numpy as np
import torch
from .probe_b1_dense_levels import oracle_records
from .seed_b1_binary_regions import build as base
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.no_grad()
def build(strength=6.,step=6.):
    net=base();w=net.weight['down2__pw'];b=net.bias['down2__pw'];w.zero_();b.zero_()
    for c in range(3):
        for j in (c,c+3,c+12):w[j,c,0,0]=1
    b[6:12]=64
    sigma=.125
    for stage in range(1,4):
        net.package['residual_encoding'][f'res{stage}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
        for half in (1,2):
            for kind in ('dw','pw'):
                key=f'res{stage}_conv{half}__{kind}';net.weight[key].zero_();net.bias[key].zero_()
        w,b=net.weight[f'res{stage}_conv1__dw'],net.bias[f'res{stage}_conv1__dw']
        for j in range(6):
            w[j,0,1,1]=-127*sigma/(4*strength)
            w[j,0,1 if j<3 else 2,2 if j<3 else 1]=127*sigma/(4*strength)
            b[j]=64
        for j in range(6,15):w[j,0,1,1]=1
        w,b=net.weight[f'res{stage}_conv1__pw'],net.bias[f'res{stage}_conv1__pw']
        for j in range(6):
            for k in (j,j+6):w[k,j,0,0]=2;w[k,j+6,0,0]=1;b[k]=-128
        for j in range(12,15):w[j,j,0,0]=1
        w,b=net.weight[f'res{stage}_conv2__dw'],net.bias[f'res{stage}_conv2__dw']
        for j in range(6):
            w[j,0,1,1]=.5;w[j,0,1 if j<3 else 0,0 if j<3 else 1]=-.5;b[j]=64
        for j in range(6,15):w[j,0,1,1]=1
        w,b=net.weight[f'res{stage}_conv2__pw'],net.bias[f'res{stage}_conv2__pw']
        for c in range(3):
            for j in (c,c+3):
                w[j,c+12,0,0]=1;w[j,c,0,0]=4*strength/127;w[j,c+3,0,0]=4*strength/127;b[j]=-128*4*strength/127
        for j in range(6,15):w[j,j,0,0]=1
    w,b=net.weight['up1_low_pw'],net.bias['up1_low_pw'];w.zero_();b.zero_()
    for c in range(3):w[c,c,0,0]=1
    kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for key in ('up1_high_dw','up2_low_dw','up2_high_dw'):
        net.weight[key].zero_();net.bias[key].zero_()
    for c in range(3):
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=1
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw'];w.zero_();b.zero_()
    for c,unit in enumerate((step,3.,3.)):w[c,c,0,0]=1/unit;b[c]=0 if c==0 else (-64)%unit/unit
    for c in range(3):net.weight['up2_high_dw'][c,0,1,1]=1
    h,b=net.weight['rgb_head'],net.bias['rgb_head'];h.zero_()
    con,sat=1.04,1.08
    matrix=h.new_tensor([[con,sat,-sat],[con,0,sat],[con,-sat,-sat]])
    units=h.new_tensor((step,3.,3.));phase=h.new_tensor((0.,(-64)%3,(-64)%3))
    h[:,:3,0,0]=matrix*units[None]*2
    b.copy_(-2*(matrix*(64+phase)[None]).sum(1)+h.new_tensor((-2.,0.,3.)))
    for key,w in net.weight.items():
        peak=w.abs().flatten(1).amax(1).clamp_min(1e-5)
        scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--strength',type=float,default=6.)
    ap.add_argument('--step',type=float,default=6.);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100681);dest=OUT/a.name;dest.mkdir(exist_ok=False)
    net=build(a.strength,a.step);raw=net.export();(dest/'initial.bin').write_bytes(raw)
    actual=t.r.IntegerB1(t.r.runtime_from_image(raw));d=t.data();errors=[]
    for row in d['validation']:
        x=t.r.tensor(t.r.read(row['path']))[None];ours=net(x,all_nodes=True);other,_=actual(x*128,True)
        errors.append(max(float((v-other[k]).abs().max()) for k,v in ours.items()))
    assert max(errors)==0
    params,residual=t.r.decode(raw);op,rp=oracle_records(params,residual)
    source=np.random.default_rng(81).integers(-128,128,(12,12,3),dtype=np.int16)
    ref=t.r.execute_integer(t.r.contract_guard()[0],source.tolist(),op,rp)
    got,_=actual(torch.tensor(source.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32),True)
    error=max(float((v[0].permute(1,2,0).cpu()-torch.tensor(ref['tensors'][k])).abs().max()) for k,v in got.items())
    assert error==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=residual,contract_sha256=t.r.CONTRACT_SHA,
        parameter_initialization_only=True),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview');shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(args=vars(a),all_node_error=max(errors),oracle_error=error,training_steps=0,
        added_operators=0,minimum_area_guarantee=False,converged_tv_solver=False,code_sha256=file_hash(__file__)))
    print('TV_REGION_PROBE',dest,max(errors),error,flush=True)


if __name__=='__main__':main()
