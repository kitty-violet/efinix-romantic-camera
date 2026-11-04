"""An actual B1 coarse-first contour computation, not image-space cleaning.

Explicit coefficients isolate ordering: smooth the surface first, then form
signed gradients and ridge response. No claim that these coefficients learned.
"""
import copy
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .b1_trained_image import Parameters,encode,decode,runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .train_b1_structural_gate import CROPS
from .train_b1_deep_contour import OUT as LABELS
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

OUT=LABELS.parent/'b1-coarse-first-probe-20261005-v1'
PARENT=LABELS.parent/'b1-feature-flow-repair-20261005-v1/reconnected_001000/coefficients-padded.bin'
GAUSS=(1,2,1,2,4,2,1,2,1);IDENT=(0,0,0,0,2,0,0,0,0)


def build(params,threshold,gradient_first=False):
    p=copy.deepcopy(params)
    def put(name,w,b,m):
        old=p[name];p[name]=Parameters(tuple(map(tuple,np.asarray(w).tolist())),tuple(np.asarray(b).tolist()),tuple(np.asarray(m).tolist()),old.relu)
    old=p['down1.pw'];w=np.array(old.weights);b=np.array(old.bias);m=np.array(old.multiplier)
    for c in range(6,16):w[c]=0;w[c,:3]=[1,2,1];b[c]=0;m[c]=32768
    put('down1.pw',w,b,m)
    old=p['down2.dw'];w=np.array(old.weights);b=np.array(old.bias);m=np.array(old.multiplier)
    for c in range(6,16):w[c]=GAUSS;b[c]=0;m[c]=8192
    if gradient_first:
        # Keep derivative precision before repeated INT8 averaging. Separate
        # positive/negative lobes stay separate until both smoothing stages.
        gx=np.array([-1,0,1,-2,0,2,-1,0,1])*2
        gy=gx.reshape(3,3).T.flatten()
        for c,k in enumerate((gx,-gx,gy,-gy),6):w[c]=k;m[c]=65536
    put('down2.dw',w,b,m)
    old=p['down2.pw'];w=np.array(old.weights);b=np.array(old.bias);m=np.array(old.multiplier)
    for c in range(6,32):
        w[c]=0;b[c]=0;m[c]=65536
        if c<24:w[c,6+(c-6)%4 if gradient_first else 6]=2
    put('down2.pw',w,b,m)
    # Keep an untouched luminance copy in12..23 to cancel the residual skip.
    for name in ('res1_conv1.dw','res1_conv2.dw'):
        old=p[name];w=np.array(old.weights);b=np.zeros(32,dtype=int);m=np.full(32,65536)
        w[6:24]=np.array(IDENT)
        for c in range(6,10 if gradient_first else 12):w[c]=GAUSS;m[c]=8192
        put(name,w,b,m)
    old=p['res1_conv1.pw'];w=np.zeros_like(np.array(old.weights));b=np.zeros(32,dtype=int);m=np.full(32,65536)
    for c in range(6,24):w[c,c]=2
    put('res1_conv1.pw',w,b,m)
    w[:]=0
    for c in range(6,24):
        w[c,6+(c-6)%4 if gradient_first else 6]=2
        w[c,10+(c-6)%4 if gradient_first else 12]=-2
    put('res1_conv2.pw',w,b,m)
    # Now detect derivatives on the coarse surface, preserving both signs.
    gx=np.array((0,0,0,-2,0,2,0,0,0));gy=np.array((0,-2,0,0,0,0,0,2,0))
    w=np.zeros((32,9),dtype=int);b=np.zeros(32,dtype=int);m=np.full(32,65536)
    for c,k in enumerate((gx,-gx,gy,-gy),6):w[c]=k
    if gradient_first:
        for c in range(6,10):w[c]=IDENT
    put('res2_conv1.dw',w,b,m)
    w=np.zeros((32,32),dtype=int)
    for c in range(6,24):w[c,6:10]=2
    if gradient_first:
        w[:]=0
        for c in range(6,24):
            d=(c-6)%4;w[c,6+d]=2;w[c,6+(d^1)]=-2
    put('res2_conv1.pw',w,b,m)
    w=np.zeros((32,9),dtype=int)
    for c in range(6,24):w[c]=(0,-1,0,-1,4,-1,0,-1,0)
    put('res2_conv2.dw',w,b,m)
    w=np.zeros((32,32),dtype=int)
    for c in range(24,32):
        if gradient_first:w[c,6:10]=4;b[c]=-4*threshold
        else:w[c,6]=8;b[c]=-8*threshold
    put('res2_conv2.pw',w,b,m)
    return p


@torch.inference_mode()
def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--gradient-first',action='store_true');args=parser.parse_args()
    if args.gradient_first:OUT=OUT.with_name('b1-coarse-gradient-precision-20261005-v2')
    configure_workspace();set_reproducible(2026100514);contract,_=contract_guard();OUT.mkdir(exist_ok=False)
    params,residual=decode(PARENT.read_bytes());cases=[r for r in json.loads((LABELS/'data.json').read_text(encoding='utf-8')) if r['split']!='train']
    base=IntegerB1(runtime_from_image(PARENT.read_bytes()));images={};records={}
    for r in cases:
        a=np.array(Image.open(r['source']));x=torch.tensor(a.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
        nodes,_=base(x,True);images[(r['id'],'source')]=Image.fromarray(a)
        images[(r['id'],'old')]=show(nodes['res3_conv1.dw'][0,24].cpu().numpy())
    thresholds=(4,8,12) if args.gradient_first else (1,2,3)
    for threshold in thresholds:
        key=f't{threshold}';dest=OUT/key;dest.mkdir();p=build(params,threshold,args.gradient_first)
        raw=encode(p,residual);raw+=bytes(20480-len(raw));(dest/'coefficients-padded.bin').write_bytes(raw)
        net=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
        op,rp=oracle_records(p);sample=np.random.default_rng(17).integers(-128,128,(12,12,3),dtype=np.int16)
        reference=execute_integer(contract,sample.tolist(),op,rp);tensors,_=net(torch.tensor(sample.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        errors={k:int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(reference['tensors'][k])).max()) for k,v in tensors.items()}
        assert max(errors.values())==0
        for r in cases:
            a=np.array(Image.open(r['source']));x=torch.tensor(a.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
            nodes,_=net(x,True);old,_=base(x,True)
            # The colour carrier is unchanged up to the colour-update residual.
            assert torch.equal(nodes['res2_add'][:,:6],old['res2_add'][:,:6])
            g=nodes['res3_conv1.dw'][0,24].cpu().numpy();im=show(g);im.save(dest/(r['id']+'_gate.png'));images[(r['id'],key)]=im
            if threshold==thresholds[0]:
                coarse=nodes['res1_add'][0,6].cpu().numpy()
                assert torch.equal(nodes['res1_add'][:,6:7],nodes['res1_conv2.dw'][:,6:7])
                Image.fromarray(np.clip(coarse*2,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).save(dest/(r['id']+'_coarse_feature.png'))
        records[key]=dict(independent_oracle_errors=errors,sha256=file_hash(dest/'coefficients-padded.bin'),threshold=threshold,
            constructed_coefficients=True,learned=False,unchanged_graph=True,external_cleanup=False,visual_accepted=False)
        print('COARSE_FIRST',key,flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(2100,1256),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,_) in enumerate(CROPS):
        variants=[(f't{t}',('保留梯度精度 ' if args.gradient_first else '粗结构后找线 ')+f'T{t}') for t in thresholds]
        for j,(col,title) in enumerate([('source','原片'),('old','原学习轮廓')]+variants):
            board.paste(images[(key,col)].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),title,font=font,fill='white')
    board.save(OUT/'actual_network_contours.png');write_json(OUT/'verification.json',records)
    shutil.copy2(__file__,OUT/Path(__file__).name)


def show(a):
    return Image.fromarray(np.clip(a*255/24,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).convert('RGB')


if __name__=='__main__':main()
