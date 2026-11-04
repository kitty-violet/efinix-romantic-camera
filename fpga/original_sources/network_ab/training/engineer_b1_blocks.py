"""Construct explicit surface shrinkage + colour levels using unchanged B1 ops.

These are designed integer coefficients, not claimed to be learned weights.
There is no operation after rgb_head except the existing display scaler.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from network_ab.contract_reference import execute_integer
from .runtime import configure_workspace, contract_guard, write_json, file_hash
from .engine import set_reproducible
from .b1_trained_image import encode, decode, runtime_from_image, Parameters
from .probe_b1_cartoon_int8 import IntegerB1
from .fpga_video_resize import bilinear_2to3

ROOT=Path(r'E:\fpga\model-training-workspace\runs')
OUT=ROOT/'b1-engineered-blocks-20261005-v1'
BASE=ROOT/'b1-deploy-preview-20261005-v1'
PRESETS={
    'mild':dict(thresholds=[2,3,3],levels=[4,2,2],contrast=1.,saturation=1.125),
    'blocks':dict(thresholds=[4,6,6],levels=[6,3,3],contrast=1.0625,saturation=1.125),
    'coarse':dict(thresholds=[6,8,8],levels=[8,4,4],contrast=1.0625,saturation=1.125),
}


def build(cfg):
    _,plan=contract_guard();params={};arrays={}
    for r in plan['nodes']:
        if r['kernel']==0:continue
        ic=1 if r['kind']=='dw3x3' else r['input_channels']
        w=np.zeros((r['channels'],r['kernel'],r['kernel'],ic),np.int32)
        b=np.zeros(r['channels'],np.int64);m=np.full(r['channels'],65536,np.int64)
        arrays[r['id']]=(w,b,m)
    def pw(name,mapping,bias=None,mul=65536):
        w,b,m=arrays[name];m[:]=mul
        for oc,terms in mapping.items():
            for ic,v in terms.items():w[oc,0,0,ic]=v
        if bias:
            for oc,v in bias.items():b[oc]=v
    def dw(name,channels=None,blur=False):
        w,b,m=arrays[name]
        channels=range(len(w)) if channels is None else channels
        for c in channels:
            if blur:w[c,:,:,0]=np.array([[1,2,1],[2,4,2],[1,2,1]]);m[c]=8192
            else:w[c,1,1,0]=2;m[c]=65536
    # Unsigned half-range RGB, then YCoCg with centred chroma.
    w,b,m=arrays['stem']
    for c in range(3):w[c,1,1,c]=1;b[c]=128
    dw('down1.dw',range(3),True)
    matrix=[[1,2,1],[2,0,-2],[-1,2,-1]]
    pw('down1.pw',{j:{c:v for c,v in enumerate(matrix[j%3]) if v} for j in range(16)},
       {j:256 if j%3 else 0 for j in range(16)},32768)
    dw('down2.dw',blur=True)
    pw('down2.pw',{j:{j%3:2} for j in range(32)})
    # Each residual executes s + soft_threshold(x-s,t). Broad edges survive;
    # weak details are replaced by the local surface estimate s.
    for stage,t in enumerate(cfg['thresholds'],1):
        prefix=f'res{stage}_conv'
        dw(prefix+'1.dw',range(3));dw(prefix+'1.dw',range(3,6),True)
        mapping={};bias={}
        for c in range(3):
            mapping[c]={3+c:2}
            mapping[3+c]={c:2,3+c:-2};bias[3+c]=-2*t
            mapping[6+c]={c:-2,3+c:2};bias[6+c]=-2*t
            mapping[9+c]={c:2}
        pw(prefix+'1.pw',mapping,bias)
        dw(prefix+'2.dw',range(12))
        pw(prefix+'2.pw',{j:{j%3:2,3+j%3:2,6+j%3:-2,9+j%3:-2} for j in range(32)})
    dw('up1_low_dw')
    pw('up1_low_pw',{j:{j%3:2} for j in range(16)})
    dw('up1_high_dw',blur=True);dw('up2_low_dw')
    levels=np.array(cfg['levels']);phase=(-64)%levels;phase[0]=0
    w,b,m=arrays['up2_low_pw']
    for c in range(3):w[c,0,0,c]=1;b[c]=phase[c];m[c]=int(np.floor(131072/levels[c]+.5))
    dw('up2_high_dw',range(3))
    # Restore colour levels and apply a fixed, explicit contrast/saturation matrix.
    con,sat=cfg['contrast'],cfg['saturation']
    matrix=np.array([[con,sat,-sat],[con,0,sat],[con,-sat,-sat]])
    w,b,m=arrays['rgb_head'];m[:]=32768
    for c in range(3):
        w[c,0,0,:3]=np.floor(8*matrix[c]*levels+.5).astype(int)
        b[c]=int(np.floor(-np.sum(w[c,0,0,:3]*(64+phase)/levels)+4*[-2,0,2][c]+.5))
    for r in plan['nodes']:
        name=r['id']
        if name not in arrays:continue
        w,b,m=arrays[name]
        assert np.max(np.abs(w))<=127 and np.max(np.abs(b))<2**31
        assert np.all((m>0)&(m<131072))
        params[name]=Parameters(tuple(tuple(int(v) for v in a) for a in w.reshape(len(w),-1)),
             tuple(int(v) for v in b),tuple(int(v) for v in m),r['activation']=='relu')
    residual={f'res{i}_add':dict(main_scaled=True,multiplier=131071,post_multiplier=131071) for i in range(1,4)}
    return params,residual


def oracle_records(params):
    _,plan=contract_guard();scales={'input':Fraction(1,128)};out={};res={}
    ratio=lambda x:[x.numerator,x.denominator]
    for r in plan['nodes']:
        name=r['id'];si=scales[r['inputs'][0]]
        if r['kind']=='nearest_x2':scales[name]=si;continue
        if r['kind']=='residual_add':
            assert si==scales[r['inputs'][1]]
            so=si*Fraction(131072,131071)
            res[name]=dict(main_scale=ratio(si),skip_scale=ratio(si),common_scale=ratio(si),output_scale=ratio(so))
        else:
            p=params[name];so=Fraction(1,128) if name=='rgb_head' else si
            out[name]=dict(weights=[list(a) for a in p.weights],bias=list(p.bias),multipliers=list(p.multiplier),
                input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(Fraction(m,131072)*so/si) for m in p.multiplier])
        scales[name]=so
    return out,res


def build_firm():
    """Firm detail removal: retain large boundaries without subtracting t."""
    p,res=build(dict(thresholds=[4,6,6],levels=[6,3,3],contrast=1.,saturation=1.))
    def point(name,mapping,bias,n):
        w=np.zeros((n,32 if name.startswith('res') or name=='up1_low_pw' else 16),int)
        b=np.zeros(n,int)
        for oc,terms in mapping.items():
            for ic,value in terms.items():w[oc,ic]=value
        for oc,value in bias.items():b[oc]=value
        p[name]=Parameters(tuple(tuple(int(v) for v in a) for a in w),tuple(int(v) for v in b),
            tuple([65536]*n),not name.endswith('2.pw'))
    for stage,t in enumerate([4,6,6],1):
        mapping={};bias={}
        for c in range(3):
            mapping[c]={3+c:2}
            for slot,sign,threshold in [(3,1,t),(6,-1,t),(9,1,2*t),(12,-1,2*t)]:
                mapping[slot+c]={c:2*sign,3+c:-2*sign};bias[slot+c]=-2*threshold
            mapping[15+c]={c:2}
        point(f'res{stage}_conv1.pw',mapping,bias,32)
        kernels=tuple((0,0,0,0,2,0,0,0,0) if c<18 else tuple([0]*9) for c in range(32))
        p[f'res{stage}_conv2.dw']=Parameters(kernels,tuple([0]*32),tuple([65536]*32),True)
        point(f'res{stage}_conv2.pw',{j:{j%3:2,3+j%3:4,6+j%3:-4,9+j%3:-2,12+j%3:2,15+j%3:-2} for j in range(32)}, {},32)
    # Explicit highlight/shadow hinge basis; colour fit is only 16 -> 3 affine.
    point('up1_low_pw',{0:{0:2},1:{1:2},2:{2:2},3:{0:2},4:{0:-2}}, {3:-192,4:96},16)
    kernel=(0,1,0,1,4,1,0,1,0)
    p['up1_high_dw']=Parameters(tuple(kernel if c<5 else tuple([0]*9) for c in range(16)),tuple([0]*16),tuple([16384]*16),True)
    return p,res


def build_contour_guided(threshold=6, gain=4):
    """Freeze a common coarse luminance contour before any surface merging.

    E = relu(|Gx|+|Gy|-threshold)*gain, from smoothed source luminance.
    Each residual: x <- x - relu(x-mean(x)-E) + relu(mean(x)-x-E).
    E=0 merges weak texture; E>=|x-mean(x)| preserves the boundary exactly.
    Same E is used for Y, Co and Cg, never one independently estimated edge/channel.
    """
    p,res=build_firm()
    matrix=[[1,2,1],[2,0,-2],[-1,2,-1]]
    w=[];bias=[]
    for j in range(16):
        row=[0]*8;kind=j%3 if j<6 else 0
        row[:3]=matrix[kind];w.append(tuple(row));bias.append(256 if kind else 0)
    p['down1.pw']=Parameters(tuple(w),tuple(bias),tuple([32768]*16),True)
    gaussian=(1,2,1,2,4,2,1,2,1)
    gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernels=[gaussian]*16;mult=[8192]*16
    for c,k in enumerate([gx,-gx,gx.T,-gx.T],6):
        kernels[c]=tuple(int(v) for v in k.ravel());mult[c]=16384
    p['down2.dw']=Parameters(tuple(kernels),tuple([0]*16),tuple(mult),True)
    w=[];bias=[]
    for j in range(32):
        row=[0]*16
        if j<24:row[j%3]=2;bias.append(0)
        else:
            for c in range(6,10):row[c]=2*gain
            bias.append(-2*gain*threshold)
        w.append(tuple(row))
    p['down2.pw']=Parameters(tuple(w),tuple(bias),tuple([65536]*32),True)
    for stage in range(1,4):
        name=f'res{stage}_conv1.dw';old=p[name];kernels=list(old.weights);mul=list(old.multiplier)
        kernels[24]=tuple([1]*9);mul[24]=32768
        p[name]=Parameters(tuple(kernels),old.bias,tuple(mul),True)
        w=np.zeros((32,32),int)
        for c in range(3):
            w[c,c]=2;w[c,3+c]=-2;w[c,24]=-2
            w[3+c,c]=-2;w[3+c,3+c]=2;w[3+c,24]=-2
        p[f'res{stage}_conv1.pw']=Parameters(tuple(tuple(int(v) for v in row) for row in w),tuple([0]*32),tuple([65536]*32),True)
        kernels=tuple((0,0,0,0,2,0,0,0,0) if c<6 else tuple([0]*9) for c in range(32))
        p[f'res{stage}_conv2.dw']=Parameters(kernels,tuple([0]*32),tuple([65536]*32),True)
        w=np.zeros((32,32),int)
        for j in range(24):w[j,j%3]=-2;w[j,3+j%3]=2
        p[f'res{stage}_conv2.pw']=Parameters(tuple(tuple(int(v) for v in row) for row in w),tuple([0]*32),tuple([65536]*32),False)
    return p,res


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100502);contract,_=contract_guard()
    OUT.mkdir(exist_ok=False)
    data=json.loads((BASE/'data.json').read_text(encoding='utf-8'));cases=data['validation']
    board=Image.new('RGB',(420*(2+len(PRESETS)),266*len(cases)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);results=[]
    for i,r in enumerate(cases):
        for j,(p,label) in enumerate([(r['source'],'原片'),(r['target'],'冻结目标')]):
            board.paste(Image.open(p).resize((420,236)),(420*j,266*i+30));draw.text((420*j+4,266*i+3),label,font=font,fill='white')
    for column,(name,cfg) in enumerate(PRESETS.items(),2):
        dest=OUT/name;dest.mkdir();parameters,residual=build(cfg)
        raw=encode(parameters,residual);padded=raw+bytes(20480-len(raw));assert decode(padded)==(parameters,residual)
        (dest/'coefficients.bin').write_bytes(raw);(dest/'coefficients-padded.bin').write_bytes(padded)
        (dest/'load_words.hex').write_text(''.join(f'{int.from_bytes(padded[i:i+8],"little"):016x}\n' for i in range(0,len(padded),8)))
        net=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
        op,rp=oracle_records(parameters);rng=np.random.default_rng(20261005)
        tests=[]
        for x in [rng.integers(-128,128,(8,12,3),dtype=np.int16),np.full((8,8,3),-128,np.int16),np.full((8,8,3),127,np.int16)]:
            ref=execute_integer(contract,x.tolist(),op,rp)
            values,_=net(torch.tensor(x.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
            errors={k:int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in values.items()}
            assert max(errors.values())==0;tests.append(errors)
        checks=[]
        for i,r in enumerate(cases):
            rgb=np.array(Image.open(r['source']).convert('RGB'));x=torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda')
            q=net(x);output=(q[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
            Image.fromarray(output).save(dest/(r['id']+'_720.png'))
            Image.fromarray(bilinear_2to3(output)).save(dest/(r['id']+'_1080.png'))
            board.paste(Image.fromarray(output).resize((420,236)),(420*column,266*i+30))
            draw.text((420*column+4,266*i+3),'直接构造：'+name,font=font,fill='white')
            delta=np.abs(output.astype(float)-np.array(Image.open(r['target'])))
            checks.append(dict(id=r['id'],mae_target=float(delta.mean())))
        write_json(dest/'verification.json',dict(status='ENGINEERED_B1_INTEGER_ORACLE_PASS',preset=cfg,checks=checks,oracle=tests,
            sha256=file_hash(dest/'coefficients-padded.bin'),transport_bytes=20480,learned=False,designed_coefficients=True,
            graph_unchanged=True,postprocess=False,actual_board=False,visual_accepted=False))
        results.append(dict(name=name,config=cfg,checks=checks));print('ENGINEERED',name,checks,flush=True)
    board.save(OUT/'comparison.jpg',quality=96)
    shutil.copy2(__file__,OUT/Path(__file__).name);write_json(OUT/'results.json',results)


if __name__=='__main__':main()
