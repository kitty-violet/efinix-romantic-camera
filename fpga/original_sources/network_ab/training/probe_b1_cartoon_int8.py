"""Calibrated INT8/Q17 evidence for current B1; never writes production files.

Integer convolution dots are computed in strict FP32 (integer operands, dot bound
<2**24); bias, Q17 product and rounding use float64 exact integers (<2**53).
The full small graph is compared node-by-node to the existing stdlib oracle.
This is a PTQ probe and runtime-parameter export, NOT an FPGA release image.
"""
import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_cartoon_budget import RUN
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .publish_b1_cartoon_budget import load
from network_ab.contract_reference import q17_multiplier,execute_integer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'hw/efinity/prototypes/frame_execution'))
sys.path.insert(0,str(ROOT/'network_ab'))
from residual_q17_align import encode_alignment


def ratio(v):
    f=Fraction(float(v));return [f.numerator,f.denominator]


def half(x):return torch.sign(x)*torch.floor(torch.abs(x)+.5)


class IntegerB1:
    def __init__(self,package,device='cuda'):
        self.package=package;self.device=device;self.plan=contract_guard()[1];self.ops={}
        for r in self.plan['nodes']:
            if r['id'] not in package['parameters']:continue
            p=package['parameters'][r['id']]
            a=np.array(p['weights'],dtype=np.float32)
            if r['kind']=='dw3x3':a=a.reshape(r['channels'],1,3,3)
            else:a=a.reshape(r['channels'],r['kernel'],r['kernel'],r['input_channels']).transpose(0,3,1,2)
            self.ops[r['id']]=(torch.tensor(a,device=device),torch.tensor(p['bias'],device=device,dtype=torch.float64)[None,:,None,None],
                torch.tensor(p['multipliers'],device=device,dtype=torch.float64)[None,:,None,None])

    @torch.inference_mode()
    def __call__(self,x,all_nodes=False):
        assert x.dtype==torch.float32 and torch.equal(x,x.round()) and x.min()>=-128 and x.max()<=127
        values={'input':x};stats={}
        for row in self.plan['nodes']:
            name=row['id'];a=values[row['inputs'][0]]
            if name in self.ops:
                w,b,m=self.ops[name]
                if row['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                dot=F.conv2d(a,w,None,stride=row['stride'],padding=0,groups=row['groups'])
                assert torch.equal(dot,dot.round())
                accum=(dot.double()+b).clamp(-(2**31),2**31-1)
                y=half(accum*m/131072)
                stats[name]=float(((y < -128)|(y >127)).float().mean())
                y=y.clamp(-128,127).float()
                if row['activation']=='relu':y=y.clamp_min(0)
            elif row['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                z=values[row['inputs'][1]];p=self.package['residual_encoding'][name]
                scaled=a if p['main_scaled'] else z
                other=z if p['main_scaled'] else a
                aligned=half(scaled.double()*p['multiplier']/131072).clamp(-128,127)
                y=half((aligned+other.double())*p['post_multiplier']/131072).clamp(-128,127).float()
            values[name]=y
        return (values,stats) if all_nodes else values['rgb_head']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,default=2000);args=ap.parse_args()
    root=configure_workspace();set_reproducible(2026100407);contract,plan=contract_guard();run=root/'runs'/RUN
    out=run/f'int8-probe-{args.step}';out.mkdir(exist_ok=False)
    path=run/f'final-{args.step}/b1_cartoon_fp32.pt';model=load(path)
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));groups={}
    for row in manifest['pairs']:groups.setdefault(row['source'],[]).append(row)
    selected=[]
    for rows in groups.values():selected.extend([rows[0],rows[len(rows)//2],rows[-1]])
    calibration=[r['current'] for r in selected];maxima={}
    with torch.inference_mode():
        for i,p in enumerate(calibration):
            values=model(as_tensor(read_image(p))[None].cuda(),return_tensors=True)
            for k,v in values.items():maxima[k]=max(maxima.get(k,0.),float(v.abs().max()))
            print('B1_INT8_CALIBRATION',i+1,len(calibration),flush=True)
    folded=copy.deepcopy(model)
    for seq in folded.operators.values():
        for i,m in enumerate(seq):
            if isinstance(m,torch.nn.BatchNorm2d):seq[i-1]=torch.nn.utils.fuse_conv_bn_eval(seq[i-1],m);seq[i]=torch.nn.Identity()
    scales={'input':1/128};params={};residual={};encoding={}
    for row in plan['nodes']:
        name=row['id'];si=scales[row['inputs'][0]]
        if row['kind']=='nearest_x2':scales[name]=si;continue
        if row['kind']=='residual_add':
            skip=scales[row['inputs'][1]];common=max(si,skip)
            so=max(maxima[name]*1.02/127,common*1.001)
            p=dict(main_scale=ratio(si),skip_scale=ratio(skip),common_scale=ratio(common),output_scale=ratio(so))
            residual[name]=p;encoding[name]=encode_alignment(p);scales[name]=so;continue
        conv=next(m for m in folded.operators[folded.names[name]] if isinstance(m,torch.nn.Conv2d))
        w=conv.weight.detach().cpu().double().numpy();b=conv.bias.detach().cpu().double().numpy()
        so=1/128 if name=='rgb_head' else max(maxima[name]*1.02/127,1e-7)
        sw=np.maximum(np.max(np.abs(w),axis=(1,2,3))/127,1e-9)
        # Keep tiny/dead channels numerically representable without Q17 m=0.
        sw=np.maximum(sw,so/(si*65536));sw=np.maximum(sw,np.abs(b)/(si*(2**29)))
        if name!='rgb_head':so=max(so,float((si*sw).max())*1.001)
        multipliers=[q17_multiplier(Fraction(si),Fraction(float(v)),Fraction(so)) for v in sw]
        assert all(0<m<131072 for m in multipliers)
        quant=lambda a:np.sign(a)*np.floor(np.abs(a)+.5)
        wi=quant(w/sw[:,None,None,None]).clip(-127,127).astype(np.int32)
        bi=quant(b/(si*sw)).astype(np.int64);assert np.abs(bi).max()<2**31
        flat=wi.transpose(0,2,3,1).reshape(row['channels'],-1).tolist()
        params[name]=dict(weights=flat,bias=bi.tolist(),multipliers=multipliers,input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(v) for v in sw])
        scales[name]=so
    package=dict(schema='trained-b1-int8-q17-probe-v1',parameters=params,residual_parameters=residual,residual_encoding=encoding,
        calibrated_on=calibration,fp32_sha256=file_hash(path),contract_file_sha256=file_hash(ROOT/'network_ab/model_b_contract.json'),
        calibration='Train-source maxima plus2percent; residual common=max, explicit legal Q17 encoding',production_release=False)
    write_json(out/'runtime_parameters.json',package)
    integer=IntegerB1(package);rng=np.random.default_rng(20261004);tests=[]
    # Cover tiny shapes, actual natural crop and saturated signed samples.
    samples=[rng.integers(-128,128,(8,12,3),dtype=np.int16),np.array(read_image(calibration[0])[:12,:16],dtype=np.int16)-128,
        np.full((8,8,3),127,dtype=np.int16),np.full((8,8,3),-128,dtype=np.int16)]
    for index,a in enumerate(samples):
        ref=execute_integer(contract,a.tolist(),params,residual)
        values,_=integer(torch.tensor(a.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        diffs={}
        for k,v in values.items():
            d=int(np.abs(v[0].permute(1,2,0).cpu().numpy().astype(np.int64)-np.array(ref['tensors'][k])).max())
            assert d==0,(index,k,d);diffs[k]=d
        tests.append(dict(sample=index,shape=list(a.shape),node_errors=diffs))
    records=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            a=read_image(row['source']);x=as_tensor(a)[None].cuda();fp=model(x)
            v,sat=integer(torch.tensor(a.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'),True)
            y=v['rgb_head'];rgb=(y[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
            Image.fromarray(rgb).save(out/(row['id']+'.png'))
            delta=(y-fp*128).abs();records.append(dict(id=row['id'],mae_to_fp32_rgb255=float(delta.mean()),max_to_fp32_rgb255=float(delta.max()),
                node_saturation=sat,new_white_fraction=float(((rgb.min(2)>245)&(a.max(2)<220)).mean())))
    write_json(out/'verification.json',dict(status='EXACT_SMALL_GRAPH_INTEGER_ORACLE_PASS_NOT_RTL',small_graph_tests=tests,records=records,
        runtime_parameters_sha256=file_hash(out/'runtime_parameters.json'),fp32_weights_sha256=file_hash(path),calibration_images=len(calibration),
        generator_same_graph=True,hardware_file_writes=False,qat=False,
        limitations=['No RTL run, no physical timing or resource revalidation.','Existing coefficient container explicitly flags synthetic/residual-invalid; this probe does not bypass it.',
            'runtime_parameters.json contains real trained INT8 weights, INT32 bias, Q17 multipliers and residual alignment; not a production flash image.']))
    print('B1_INT8_PROBE_COMPLETE',[(r['id'],round(r['mae_to_fp32_rgb255'],3)) for r in records],flush=True)


if __name__=='__main__':main()
