"""Retain B1 features, replace final colour generation with an explicit palette.

The 16 -> YCoCg projection is a small closed-form fit on TRAINING sources.
The number/spacing of colour levels is directly specified, not learned by SGD.
All effects run inside the existing B1 graph and resident coefficient image.
"""
import copy
import argparse
import hashlib
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .b1_trained_image import encode,decode,Parameters,runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .engineer_b1_blocks import build,build_firm,build_contour_guided
from .fpga_video_resize import bilinear_2to3

ROOT=Path(r'E:\fpga\model-training-workspace\runs')
BASE=ROOT/'b1-deploy-preview-20261005-v1'
OUT=ROOT/'b1-controlled-colour-20261005-v1'
PRESETS={'fine':dict(levels=[4,2,2]),'blocks':dict(levels=[6,3,3]),'bold':dict(levels=[8,4,4])}


def ycocg(rgb):
    a=rgb.astype(np.float64)/2
    return np.stack(((a[...,0]+2*a[...,1]+a[...,2])/4,
        (a[...,0]-a[...,2])/2+64,(-a[...,0]+2*a[...,1]-a[...,2])/4+64),-1)


@torch.inference_mode()
def main(designed=False,contours=False):
    global OUT
    if designed:OUT=ROOT/'b1-controlled-firm-20261005-v1'
    if contours:OUT=ROOT/'b1-contour-gated-20261005-v1'
    configure_workspace();set_reproducible(2026100503);contract_guard();OUT.mkdir(exist_ok=True)
    data=json.loads((BASE/'data.json').read_text(encoding='utf-8'))
    if designed or contours:
        params,residual=build_contour_guided() if contours else build_firm();raw=encode(params,residual);image=raw+bytes(20480-len(raw))
    else:image=(BASE/'package-qat600/coefficients-padded.bin').read_bytes();params,residual=decode(image)
    source_hash=hashlib.sha256(image).hexdigest();(OUT/'feature_body.bin').write_bytes(image)
    feature=IntegerB1(runtime_from_image(image));rng=np.random.default_rng(20261005)
    xrows=[];yrows=[]
    cached=OUT/'projection.json'
    for index,row in enumerate([] if cached.exists() else data['pairs']):
        rgb=np.array(Image.open(row['current']));target=np.array(Image.open(row['current_target']))
        x=torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda')
        nodes,_=feature(x,True);f=nodes['up2_low_dw'][0].permute(1,2,0).cpu().numpy().astype(float)
        target=target.reshape(360,2,640,2,3).mean((1,3));y=ycocg(target)
        picks=rng.choice(f.shape[0]*f.shape[1],12000,replace=False)
        xrows.append(f.reshape(-1,16)[picks]);yrows.append(y.reshape(-1,3)[picks])
        print('PROJECT_COLOUR',index+1,len(data['pairs']),flush=True)
    if cached.exists():
        fit=json.loads(cached.read_text());assert fit['source_package_sha256']==source_hash
        assert fit['data_sha256']==file_hash(BASE/'data.json')
        matrix=np.array(fit['matrix']);bias=np.array(fit['bias'])
    else:
        x=np.concatenate(xrows);y=np.concatenate(yrows);xm=x.mean(0);ym=y.mean(0)
        matrix=np.linalg.solve((x-xm).T@(x-xm)+np.eye(16)*100,(x-xm).T@(y-ym))
        bias=ym-xm@matrix
        write_json(cached,dict(matrix=matrix.tolist(),bias=bias.tolist(),samples=len(x),
            train_mae=np.mean(np.abs(x@matrix+bias-y),axis=0).tolist(),source_package_sha256=source_hash,
            data_sha256=file_hash(BASE/'data.json'),train_sources=[r['current'] for r in data['pairs']],validation_fit=False))
    cases=data['validation'];cols=2+len(PRESETS)
    board=Image.new('RGB',(420*cols,266*len(cases)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for i,r in enumerate(cases):
        for j,(p,label) in enumerate([(r['source'],'原片'),(r['target'],'冻结目标')]):
            board.paste(Image.open(p).resize((420,236)),(420*j,266*i+30));draw.text((420*j+4,266*i+3),label,font=font,fill='white')
    for column,(name,cfg) in enumerate(PRESETS.items(),2):
        dest=OUT/name;dest.mkdir(exist_ok=True);p=copy.deepcopy(params)
        levels=np.array(cfg['levels']);phase=(-64)%levels;phase[0]=0
        weights=[];biases=[];mult=[]
        for c in range(8):
            if c<3:
                scale=np.max(np.abs(matrix[:,c]))/127
                w=np.sign(matrix[:,c])*np.floor(np.abs(matrix[:,c])/scale+.5)
                b=int(np.sign(bias[c]+phase[c])*np.floor(abs(bias[c]+phase[c])/scale+.5))
                m=int(np.floor(scale/levels[c]*131072+.5));assert 0<m<131072
                weights.append(tuple(int(v) for v in w));biases.append(b);mult.append(m)
            else:weights.append(tuple([0]*16));biases.append(0);mult.append(65536)
        p['up2_low_pw']=Parameters(tuple(weights),tuple(biases),tuple(mult),True)
        colour_decoder,_=build(dict(thresholds=[0,0,0],levels=cfg['levels'],contrast=1.,saturation=1.))
        # Palette indices stay discrete through a small interpolation stencil.
        kernel=(1,2,1,2,4,2,1,2,1)
        p['up2_high_dw']=Parameters(tuple(kernel if c<3 else tuple([0]*9) for c in range(8)),tuple([0]*8),tuple([8192]*8),True)
        head=colour_decoder['rgb_head'];p['rgb_head']=Parameters(head.weights,tuple(head.bias[c]-4*[-2,0,2][c] for c in range(3)),head.multiplier,False)
        raw=encode(p,residual);padded=raw+bytes(20480-len(raw))
        (dest/'coefficients.bin').write_bytes(raw);(dest/'coefficients-padded.bin').write_bytes(padded)
        (dest/'load_words.hex').write_text(''.join(f'{int.from_bytes(padded[i:i+8],"little"):016x}\n' for i in range(0,len(padded),8)))
        assert decode(padded)==(p,residual)
        net=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()));checks=[]
        for i,r in enumerate(cases):
            rgb=np.array(Image.open(r['source']));x=torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda')
            if contours:
                nodes,_=net(x,True);q=nodes['rgb_head']
                contour=nodes['down2.pw'][0,24].cpu().numpy().astype(np.uint8)
                Image.fromarray(np.minimum(contour.astype(int)*4,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).save(dest/(r['id']+'_contour.png'))
            else:q=net(x)
            out=(q[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
            Image.fromarray(out).save(dest/(r['id']+'_720.png'));Image.fromarray(bilinear_2to3(out)).save(dest/(r['id']+'_1080.png'))
            board.paste(Image.fromarray(out).resize((420,236)),(420*column,266*i+30));draw.text((420*column+4,266*i+3),'受控颜色：'+name,font=font,fill='white')
            checks.append(dict(id=r['id'],mae_target=float(np.abs(out.astype(float)-np.array(Image.open(r['target']))).mean())))
        write_json(dest/'verification.json',dict(status='BYTE_ROUNDTRIP_PASSED_VISUAL_CANDIDATE',config=cfg,
            checks=checks,sha256=file_hash(dest/'coefficients-padded.bin'),transport_bytes=20480,
            graph_unchanged=True,manual_output_controls=True,designed_surface=designed or contours,contour_guided=contours,extra_postprocess=False,actual_board=False))
        print('CONTROLLED',name,checks,flush=True)
    board.save(OUT/'comparison.jpg',quality=96);shutil.copy2(__file__,OUT/Path(__file__).name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--designed',action='store_true');parser.add_argument('--contours',action='store_true');args=parser.parse_args();main(args.designed,args.contours)
