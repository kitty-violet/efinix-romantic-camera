"""Separate contour errors from a blurry decoder; test actual B1 coefficients.

The oracle-contour intervention is diagnostic only and is never packaged.
Decoder candidates consume the learned contour and are ordinary 20 KiB images.
"""
import copy
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .b1_trained_image import Parameters,encode,decode,runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1,half
from .train_b1_structural_gate import CROPS
from .train_b1_deep_contour import OUT as LABELS
from .resume_b1_contour_precision import OUT as PARENT
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

OUT=PARENT.parent/'b1-surface-path-audit-20261005-v1'


@torch.inference_mode()
def diagnostic_forward(net,x,gate=None):
    values={'input':x}
    for row in net.plan['nodes']:
        name=row['id'];a=values[row['inputs'][0]]
        if name in net.ops:
            w,b,m=net.ops[name]
            if row['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
            dot=F.conv2d(a,w,None,stride=row['stride'],groups=row['groups'])
            y=half((dot.double()+b).clamp(-(2**31),2**31-1)*m/131072).clamp(-128,127).float()
            if row['activation']=='relu':y=y.clamp_min(0)
        elif row['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
        else:
            z=values[row['inputs'][1]];r=net.package['residual_encoding'][name]
            small,other=(a,z) if r['main_scaled'] else (z,a)
            y=half((half(small.double()*r['multiplier']/131072).clamp(-128,127)+other.double())*r['post_multiplier']/131072).clamp(-128,127).float()
        if name=='res2_add' and gate is not None:
            y=y.clone();y[:,24:32]=gate.expand(-1,8,-1,-1)
        values[name]=y
    return values


def decoder_candidate(params,amount):
    p=copy.deepcopy(params)
    # Duplicate the already merged colour, not the raw photographic detail.
    old=p['up1_low_pw'];w=np.array(old.weights);b=np.array(old.bias);m=np.array(old.multiplier)
    for c in range(3):
        for slot in (6+c,9+c):
            w[slot]=0;w[slot,c]=2;b[slot]=0;m[slot]=65536
    p['up1_low_pw']=Parameters(tuple(map(tuple,w.tolist())),tuple(b.tolist()),tuple(m.tolist()),old.relu)
    old=p['up1_high_dw'];w=list(old.weights);b=list(old.bias);m=list(old.multiplier)
    # A narrower interpolation kernel and a signed, paired coarse residual.
    for c in range(5):w[c]=(0,1,0,1,12,1,0,1,0);m[c]=8192
    lap=(0,-2,0,-2,8,-2,0,-2,0)
    for c in range(3):
        w[6+c]=lap;w[9+c]=tuple(-v for v in lap)
        b[6+c]=b[9+c]=0;m[6+c]=m[9+c]=16384
    p['up1_high_dw']=Parameters(tuple(w),tuple(b),tuple(m),old.relu)
    old=p['up2_low_pw'];w=np.array(old.weights);b=np.array(old.bias);m=np.array(old.multiplier)
    levels=np.array([6,3,3]);phase=(-64)%levels;phase[0]=0
    for c in range(3):
        w[c]=0;w[c,c]=4;w[c,6+c]=round(4*amount);w[c,9+c]=-round(4*amount)
        b[c]=4*phase[c];m[c]=round(131072/(4*levels[c]))
    p['up2_low_pw']=Parameters(tuple(map(tuple,w.tolist())),tuple(b.tolist()),tuple(m.tolist()),old.relu)
    return p


def rgb(nodes):
    return Image.fromarray((nodes['rgb_head'][0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8))


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100510);contract,_=contract_guard();OUT.mkdir(exist_ok=False)
    source=PARENT/'step_002000/coefficients-padded.bin'
    raw=source.read_bytes();params,residual=decode(raw);base=IntegerB1(runtime_from_image(raw))
    nets={'learned':base};records={}
    for key,amount in [('decoder_mild',.5),('decoder_firm',1.)]:
        p=decoder_candidate(params,amount);image=encode(p,residual);image+=bytes(20480-len(image))
        dest=OUT/key;dest.mkdir();(dest/'coefficients-padded.bin').write_bytes(image)
        nets[key]=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
        op,rp=oracle_records(p);sample=np.random.default_rng(41).integers(-128,128,(12,12,3),dtype=np.int16)
        ref=execute_integer(contract,sample.tolist(),op,rp)
        nodes,_=nets[key](torch.tensor(sample.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        error=max(int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in nodes.items())
        assert error==0
        records[key]=dict(oracle_error=error,sha256=file_hash(dest/'coefficients-padded.bin'),
            graph_unchanged=True,external_postprocessing=False,learned_contour=True,designed_decoder=True)
    data=json.loads((LABELS/'data.json').read_text(encoding='utf-8'));images={}
    for r in data:
        if r['split']=='train':continue
        src=np.array(Image.open(r['source']).convert('RGB'));x=torch.tensor(src.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
        original,_=base(x,True);replicated=diagnostic_forward(base,x)
        assert max(float((v-replicated[k]).abs().max()) for k,v in original.items())==0
        t=torch.tensor(np.load(r['label']).astype(np.float32)[::4,::4],device='cuda')[None,None]
        ideal=diagnostic_forward(base,x,half(t*32-8))
        images[r['id']]=dict(source=Image.fromarray(src),teacher=Image.open(r['target']).convert('RGB'),
                             learned=rgb(original),oracle=rgb(ideal))
        for key,net in nets.items():
            if key=='learned':continue
            nodes,_=net(x,True)
            assert torch.equal(original['res3_add'],nodes['res3_add'])
            images[r['id']][key]=rgb(nodes);images[r['id']][key].save(OUT/key/(r['id']+'_raw_rgb.png'))
        print('SURFACE_AUDIT',r['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for filename,cols,titles in [
        ('oracle_diagnostic.png',['teacher','learned','oracle'],['冻结彩色目标','现有真实权重输出','仅诊断：注入正确轮廓']),
        ('actual_decoder_candidates.png',['teacher','learned','decoder_mild','decoder_firm'],['冻结彩色目标','现有真实权重输出','真实权重：重建修正 0.5','真实权重：重建修正 1.0'])]:
        board=Image.new('RGB',(420*len(cols),314*len(CROPS)),'#17202a');draw=ImageDraw.Draw(board)
        for i,(key,box,name) in enumerate(CROPS):
            for j,col in enumerate(cols):
                board.paste(images[key][col].crop(box),(420*j,314*i+34))
                draw.text((420*j+4,314*i+4),titles[j],font=font,fill='white')
        board.save(OUT/filename)
    write_json(OUT/'verification.json',dict(parent_sha256=file_hash(source),candidates=records,
        oracle_injection_is_diagnostic_only=True,visual_accepted=False))
    shutil.copy2(__file__,OUT/Path(__file__).name)


if __name__=='__main__':main()
