"""Independent region-abstraction experiments; no change to FPGA or B1.

L0 solver follows Xu et al., SIGGRAPH Asia2011 half-quadratic equations.
Rolling guidance uses a separable approximation to joint bilateral filtering,
inspired by Zhang et al., ECCV2014. This is NOT the authors' implementation.
These algorithms are CPU/GPU research references, NOT deployable B1 weights.
"""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from .runtime import configure_workspace, file_hash, write_json, contract_guard
from .engine import set_reproducible

RUN='region-mechanisms-20261004'
ROOT=Path('E:/fpga/model-training-workspace')


def tensor(a):return torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/255
def rgb(t):return (t[0].permute(1,2,0).detach().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)


def l0_surface(x,lam=.02,kappa=2.,beta_max=1e5,pad=32):
    """Global L0 gradient sparsity; FFT is explicitly outside current FPGA ops."""
    if pad:x=F.pad(x,(pad,)*4,mode='reflect')
    h,w=x.shape[-2:];yy=torch.arange(h,device=x.device);xx=torch.arange(w,device=x.device)
    denom=(4-2*torch.cos(2*math.pi*yy/h)[:,None]-2*torch.cos(2*math.pi*xx/w)[None,:]).clamp_min(0)
    original=torch.fft.fft2(x);s=x.clone();beta=2*lam
    while beta<beta_max:
        dx=torch.roll(s,-1,-1)-s;dy=torch.roll(s,-1,-2)-s
        retain=((dx.square()+dy.square()).sum(1,keepdim=True)>=lam/beta)
        dx=dx*retain;dy=dy*retain
        divergence=torch.roll(dx,1,-1)-dx+torch.roll(dy,1,-2)-dy
        s=torch.fft.ifft2((original+beta*torch.fft.fft2(divergence))/(1+beta*denom)).real
        beta*=kappa
    return s[:,:,pad:-pad,pad:-pad] if pad else s


def gaussian(x,sigma):
    radius=math.ceil(sigma*3);axis=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
    weights=torch.exp(-axis.square()/(2*sigma*sigma));weights/=weights.sum()
    c=x.shape[1]
    a=F.conv2d(F.pad(x,(radius,radius,0,0),mode='replicate'),weights[None,None,None,:].repeat(c,1,1,1),groups=c)
    return F.conv2d(F.pad(a,(0,0,radius,radius),mode='replicate'),weights[None,None,:,None].repeat(c,1,1,1),groups=c)


def joint_axis(value,guide,sigma_space,sigma_color,axis):
    radius=math.ceil(sigma_space*2);h,w=value.shape[-2:]
    pad=(radius,radius,0,0) if axis==-1 else (0,0,radius,radius)
    a,g=[F.pad(v,pad,mode='replicate') for v in (value,guide)]
    total=torch.zeros_like(value);normal=torch.zeros_like(value[:,:1])
    for d in range(-radius,radius+1):
        other=a[:,:,:,radius+d:radius+d+w] if axis==-1 else a[:,:,radius+d:radius+d+h,:]
        ng=g[:,:,:,radius+d:radius+d+w] if axis==-1 else g[:,:,radius+d:radius+d+h,:]
        weight=torch.exp(-((ng-guide).square().mean(1,keepdim=True))/(2*sigma_color*sigma_color)-d*d/(2*sigma_space*sigma_space))
        total+=other*weight;normal+=weight
    return total/normal


def rolling_surface(x,sigma_space=4.,sigma_color=.10,iterations=4):
    guide=gaussian(x,sigma_space)
    for i in range(iterations):
        axes=(-1,-2) if i%2==0 else (-2,-1)
        first=joint_axis(x,guide,sigma_space,sigma_color,axes[0])
        guide=joint_axis(first,guide,sigma_space,sigma_color,axes[1])
    return guide


def outline(surface,strength=.7):
    """Continuous contour on the simplified image; no hard Canny/palette jumps."""
    lum=(surface*surface.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    dog=gaussian(lum,.65)-gaussian(lum,1.3)
    ink=1-strength*torch.tanh(F.relu(-dog-.003)*24)
    return (surface*ink).clamp(0,1)


def apply(x,kind):
    if kind=='l0_005':return outline(l0_surface(x,.005))
    if kind=='l0_020':return outline(l0_surface(x,.020))
    if kind=='l0_050':return outline(l0_surface(x,.050))
    if kind=='rolling_4':return outline(rolling_surface(x,4.,.10,4))
    if kind=='rolling_6':return outline(rolling_surface(x,6.,.13,4))
    raise ValueError(kind)


def checks(out):
    records=[]
    with torch.no_grad():
        const=torch.ones(1,3,64,96,device='cuda')*.37
        for kind,fn in [('l0',l0_surface),('rolling',rolling_surface)]:
            error=float((fn(const)-const).abs().max());assert error<2e-5
            x=const.clone();x[:,:,:,:48]=.2;x[:,:,:,48:]=.8
            y=fn(x);contrast=float(y[:,:,:,52:80].mean()-y[:,:,:,16:44].mean())
            assert contrast>.50,(kind,contrast)
            assert torch.isfinite(y).all()
            records.append(dict(kind=kind,constant_error=error,step_contrast=contrast))
    write_json(out/'checks.json',dict(status='NUMERICAL_TESTS_PASS_NOT_STYLE_ACCEPTANCE',records=records))


def stills(run):
    manifest=json.loads((ROOT/'runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))
    config=dict(algorithms=['l0_005','l0_020','l0_050','rolling_4','rolling_6'],
        refs=['https://jiaya.me/file/archive/projects/L0smoothing/','https://www.cse.cuhk.edu.hk/~leojia/projects/rollguidance/',
              'https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/ximgproc/src/rolling_guidance_filter.cpp'],
        source_sha256=file_hash(__file__),precision='FP32',processing_resolution=[1280,720],
        deployment='None. FFT/global optimization, exponential and dynamic weighted filtering are NOT current B1 operations.',
        temporal='Stateless; no flow/no blending/no face or source paste. Temporal quality not assumed.')
    write_json(run/'config.json',config);checks(run);records=[]
    for row in manifest['validation']:
        s=np.asarray(Image.open(row['source']).convert('RGB'));x=tensor(s)
        for kind in config['algorithms']:
            out=run/kind;out.mkdir(exist_ok=True);t0=time.perf_counter()
            with torch.no_grad():z=apply(x,kind);result=rgb(z)
            torch.cuda.synchronize();Image.fromarray(result).save(out/(row['id']+'.png'))
            records.append(dict(id=row['id'],kind=kind,seconds=time.perf_counter()-t0,source_sha256=file_hash(row['source'])))
        print('REGION_STILL',row['id'],flush=True)
    write_json(run/'stills.json',dict(records=records));pictures(run)


def pictures(run):
    manifest=json.loads((ROOT/'runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    kinds=['source','teacher','l0_005','l0_020','l0_050','rolling_4','rolling_6']
    for scene,key,xy in [('stairs','val_stairs_090',(150,430)),('pillar','val_stairs_090',(424,320)),('people','val_plaza_090',(80,400))]:
        row=next(r for r in manifest['validation'] if r['id']==key);x,y=xy
        board=Image.new('RGB',(320*4,298*2),'#18202a');d=ImageDraw.Draw(board)
        for i,kind in enumerate(kinds):
            path=row['source'] if kind=='source' else row['target'] if kind=='teacher' else run/kind/(key+'.png')
            im=Image.open(path).crop((x,y,x+320,y+264));xx,yy=(i%4)*320,(i//4)*298
            d.text((xx+5,yy+5),kind,font=font,fill='white');board.paste(im,(xx,yy+30))
        board.save(run/(scene+'_mechanisms.png'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stills']);args=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100414);cv2.setNumThreads(2)
    run=ROOT/'runs'/RUN;run.mkdir(exist_ok=False);stills(run)


if __name__=='__main__':main()
