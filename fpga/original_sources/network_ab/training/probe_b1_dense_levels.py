"""Use runtime residual scales to free duplicate planes, with the fixed B1 ABI.

All 31 ordinal planes now carry colours. The old skip is scaled to zero for
INT8 values using the legal Q17 multiplier 1. This changes package metadata,
not hardware or graph. Checkpoints explicitly preserve that metadata.
"""
import argparse
from fractions import Fraction
import shutil
import torch
import numpy as np
from .seed_b1_binary_regions import build as base
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible

LEVELS=[(0,v) for v in range(8,121,8)]+[(1,v) for v in (44,49,54,59,69,74,79,84)]+[(2,v) for v in (44,49,54,59,69,74,79,84)]


def oracle_records(params,residual):
    scales={'input':Fraction(1,128)};out={};res={}
    ratio=lambda x:[x.numerator,x.denominator]
    for r in t.r.contract_guard()[1]['nodes']:
        name=r['id'];si=scales[r['inputs'][0]]
        if r['kind']=='nearest_x2':scales[name]=si;continue
        if r['kind']=='residual_add':
            sk=scales[r['inputs'][1]];common=max(si,sk);q=residual[name]
            so=common*Fraction(131072,q['post_multiplier'])
            assert (si<sk)==q['main_scaled']
            assert min(si,sk)/common==Fraction(q['multiplier'],131072)
            res[name]=dict(main_scale=ratio(si),skip_scale=ratio(sk),common_scale=ratio(common),output_scale=ratio(so))
        else:
            so=si
            if name.startswith('res') and name.endswith('conv2.pw'):
                skip='down2.pw' if name[3]=='1' else 'res'+str(int(name[3])-1)+'_add'
                q=residual[f'res{name[3]}_add']
                factor=Fraction(q['multiplier'],131072)
                so=scales[skip]*(factor if q['main_scaled'] else 1/factor)
            if name=='rgb_head':so=Fraction(1,128)
            p=params[name]
            out[name]=dict(weights=[list(a) for a in p.weights],bias=list(p.bias),multipliers=list(p.multiplier),
                input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(Fraction(m,131072)*so/si) for m in p.multiplier])
        scales[name]=so
    return out,res


@torch.no_grad()
def build(passes=2,outline=.6,sharp=.5,paired=False):
    net=base()
    for stage in range(1,4):
        net.package['residual_encoding'][f'res{stage}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    w=net.weight['down2__pw'];b=net.bias['down2__pw'];w.zero_();b.zero_()
    for j,(c,threshold) in enumerate(LEVELS):
        w[j,c,0,0]=127;w[j,c+3,0,0]=127;b[j]=-254*threshold-127
    for stage in range(1,4):
        for half in (1,2):
            key=f'res{stage}_conv{half}__dw';w,b=net.weight[key],net.bias[key];w.zero_();b.zero_()
            for c in range(31):
                if (stage-1)*2+half<=passes:w[c,0]=2;b[c]=-9*127
                else:w[c,0,1,1]=1
            key=f'res{stage}_conv{half}__pw';w,b=net.weight[key],net.bias[key];w.zero_();b.zero_()
            for c in range(31):w[c,c,0,0]=1
    # Reconstruct three colour coordinates at180p; narrower edge interpolation
    # later avoids applying a separate blur over large areas of constant colour.
    w,b=net.weight['up1_low_pw'],net.bias['up1_low_pw'];w.zero_();b.zero_()
    w[0,:15,0,0]=8/127;b[0]=4
    for row,start in ((1,15),(2,23)):
        w[row,start:start+8,0,0]=5/127
        w[row,start+3,0,0]=7.5/127;w[row,start+4,0,0]=7.5/127
        b[row]=41.5
    kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for key in ('up1_high_dw','up2_low_dw','up2_high_dw'):
        net.weight[key].zero_();net.bias[key].zero_()
    for c in range(3):
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0]=-sharp*kernel
        net.weight['up2_low_dw'][c,0,1,1]+=1+sharp
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw'];w.zero_();b.zero_()
    con,sat=1.04,1.08
    w[:3,:3,0,0]=w.new_tensor([[con,sat,-sat],[con,0,sat],[con,-sat,-sat]])
    b[:3]=w.new_tensor([64*(1-con)-1,64*(1-con-sat),64*(1-con+2*sat)+1.5])
    for c in range(3):net.weight['up2_high_dw'][c,0]=kernel
    sobel=w.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/4
    for c,k in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):
        w[c,0,0,0]=1;net.weight['up2_high_dw'][c,0]=k
    net.weight['rgb_head'].zero_()
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2
    net.weight['rgb_head'][:,3:7,0,0]=-outline
    net.bias['rgb_head'].fill_(-128)
    if paired:
        # Preserve ordinal region information until360p rather than mixing RGB
        # before upsampling. Two binary planes share one three-state channel.
        w,b=net.weight['up1_low_pw'],net.bias['up1_low_pw'];w.zero_();b.zero_()
        for c in range(8):
            for j in range(2*c,min(2*c+2,15)):w[c,j,0,0]=.5
        for axis,start in ((8,15),(12,23)):
            for c in range(4):w[axis+c,start+2*c:start+2*c+2,0,0]=.5
        for c in range(16):
            net.weight['up1_high_dw'][c,0]=kernel
            net.weight['up2_low_dw'][c].zero_()
            net.weight['up2_low_dw'][c,0,1,1]=1/64
        w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw'];w.zero_();b.zero_()
        w[:3,:8,0,0]=8*con
        w[0,8:12,0,0]=5*sat;w[2,8:12,0,0]=-5*sat
        w[0,12:16,0,0]=-5*sat;w[1,12:16,0,0]=5*sat;w[2,12:16,0,0]=-5*sat
        b[:3]=w.new_tensor([4*con+64*(1-con)-1,4*con+64*(1-con)-20*sat,4*con+64*(1-con)+40*sat+1.5])
        for c in range(3,7):w[c,:8,0,0]=8;b[c]=4
    for key,w in net.weight.items():
        peak=w.abs().flatten(1).amax(1).clamp_min(1e-5)
        scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--passes',type=int,default=2)
    ap.add_argument('--sharp',type=float,default=.5);ap.add_argument('--paired',action='store_true');args=ap.parse_args()
    configure_workspace();set_reproducible(2026100673);dest=OUT/args.name
    assert not (dest/'review.json').exists(), 'Do not overwrite a completed probe'
    dest.mkdir(exist_ok=True)
    net=build(args.passes,sharp=args.sharp,paired=args.paired);raw=net.export();(dest/'initial.bin').write_bytes(raw)
    actual=t.r.IntegerB1(t.r.runtime_from_image(raw));d=t.data();errors=[]
    for row in d['validation']:
        x=t.r.tensor(t.r.read(row['path']))[None];a=net(x,all_nodes=True);b,_=actual(x*128,True)
        errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
    assert max(errors)==0
    params,residual=t.r.decode(raw);op,rp=oracle_records(params,residual)
    # Independent rational/std-library oracle, with package-derived residual scales.
    source=np.random.default_rng(2026100673).integers(-128,128,(12,12,3),dtype=np.int16)
    reference=t.r.c.execute_integer(t.r.contract_guard()[0],source.tolist(),op,rp)
    sample=torch.tensor(source.transpose(2,0,1).copy(),device='cuda',dtype=torch.float32)[None]
    nodes,_=actual(sample,True)
    oracle_error=max(float((v[0].permute(1,2,0).cpu()-torch.tensor(reference['tensors'][k])).abs().max()) for k,v in nodes.items())
    assert oracle_error==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,
        residual_encoding=residual,parameter_initialization_only=True),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview');shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(args=vars(args),all_node_error=max(errors),oracle_error=oracle_error,
        residual_encoding=residual,training_steps=0,minimum_area_guarantee=False,added_operators=0,code_sha256=file_hash(__file__)))
    print('DENSE_LEVEL_PROBE',dest,max(errors),oracle_error,flush=True)


if __name__=='__main__':main()
