"""Contour-only diagnosis: coherent boundaries instead of raw texture edges.

No colour/video release. Compare actual B1 intermediate gates at native scale.
All cleaning below is encoded in ordinary B1 conv/ReLU/residual coefficients.
"""
import copy
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible
from .b1_trained_image import Parameters,encode,decode,runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .engineer_b1_blocks import build_contour_guided,oracle_records
from network_ab.contract_reference import execute_integer

ROOT=Path(r'E:\fpga\model-training-workspace\runs')
OUT=ROOT/'b1-structural-contour-audit-20261005-v1'
BASE=ROOT/'b1-deploy-preview-20261005-v1'
VARIANTS={'c1':dict(penalty=4,threshold=1),'c2':dict(penalty=5,threshold=1),'c3':dict(penalty=5,threshold=2)}
GAUSS=(1,2,1,2,4,2,1,2,1);IDENT=(0,0,0,0,2,0,0,0,0)


def build(cfg):
    p,res=build_contour_guided()
    def point(name,w,b=None,relu=True):
        p[name]=Parameters(tuple(tuple(int(v) for v in row) for row in w),tuple([0]*len(w) if b is None else b),tuple([65536]*len(w)),relu)
    w=np.zeros((32,16),int)
    for j in list(range(9))+list(range(18,24)):w[j,j%3]=2
    for j in range(4):w[9+j,6+j]=2;w[13+j,6+j]=2
    w[17,6:10]=2
    point('down2.pw',w)
    # Coherence compares the magnitude of the local mean signed gradient with
    # mean absolute gradient. Opposing texture directions cancel before gating.
    kernels=[tuple([0]*9)]*32;mult=[65536]*32
    for c in [9,10,11,12,17]:kernels[c]=GAUSS;mult[c]=8192
    p['res1_conv1.dw']=Parameters(tuple(kernels),tuple([0]*32),tuple(mult),True)
    w=np.zeros((32,32),int)
    w[9,9]=2;w[9,10]=-2;w[10,9]=-2;w[10,10]=2
    w[11,11]=2;w[11,12]=-2;w[12,11]=-2;w[12,12]=2;w[17,17]=2
    point('res1_conv1.pw',w)
    p['res1_conv2.dw']=Parameters(tuple(kernels),tuple([0]*32),tuple(mult),True)
    w=np.zeros((32,32),int);b=[0]*32
    for c in range(24,32):w[c,9:13]=8;w[c,17]=-cfg['penalty'];b[c]=-8*cfg['threshold']
    point('res1_conv2.pw',w,b,False)
    # ReLU the signed coherence score before it controls any colour update.
    old=p['res2_conv1.dw'];kernels=list(old.weights);mult=list(old.multiplier)
    kernels[24]=IDENT;kernels[25]=tuple(-v for v in IDENT);mult[24]=mult[25]=65536
    p['res2_conv1.dw']=Parameters(tuple(kernels),old.bias,tuple(mult),True)
    old=p['res2_conv1.pw'];w=[list(a) for a in old.weights];w[7][25]=2
    point('res2_conv1.pw',w)
    old=p['res2_conv2.dw'];kernels=list(old.weights);kernels[7]=IDENT
    p['res2_conv2.dw']=Parameters(tuple(kernels),old.bias,old.multiplier,True)
    old=p['res2_conv2.pw'];w=[list(a) for a in old.weights]
    for c in range(9,18):w[c]=[0]*32
    for c in range(24,32):w[c]=[0]*32;w[c][7]=2
    point('res2_conv2.pw',w,relu=False)
    return p,res


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100505);cv2.setNumThreads(2);contract,_=contract_guard();OUT.mkdir(exist_ok=False)
    data=json.loads((BASE/'data.json').read_text(encoding='utf-8'));cases=data['validation']
    old=IntegerB1(runtime_from_image((ROOT/'b1-contour-gated-20261005-v2/crisp/coefficients-padded.bin').read_bytes()))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    sample={};records=[]
    for r in cases:
        rgb=np.array(Image.open(r['source']));x=torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda')
        values,_=old(x,True);sample[r['id']]=(rgb,x,values['down2.pw'][0,24].cpu().numpy())
    for name,cfg in VARIANTS.items():
        dest=OUT/name;dest.mkdir();params,residual=build(cfg);raw=encode(params,residual);image=raw+bytes(20480-len(raw))
        (dest/'coefficients-padded.bin').write_bytes(image);assert decode(image)==(params,residual)
        net=IntegerB1(runtime_from_image(image));op,rp=oracle_records(params);rng=np.random.default_rng(16)
        x=rng.integers(-128,128,(12,12,3),dtype=np.int16);ref=execute_integer(contract,x.tolist(),op,rp)
        tensors,_=net(torch.tensor(x.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        errors={k:int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in tensors.items()}
        assert max(errors.values())==0
        for r in cases:
            rgb,x,old_gate=sample[r['id']];nodes,_=net(x,True);gate=nodes['res2_add'][0,24].cpu().numpy()
            # Check that this is the exact positive gate used by the first merge.
            used=nodes['res2_conv1.dw'][0,24].cpu().numpy();assert np.array_equal(gate,used) and gate.min()>=0
            im=Image.fromarray(np.clip(gate*4,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST)
            im.save(dest/(r['id']+'_gate.png'));np.save(dest/(r['id']+'_gate180.npy'),gate.astype(np.int8))
            n,l,stats,_=cv2.connectedComponentsWithStats((gate>0).astype(np.uint8),8)
            row=dict(id=r['id'],variant=name,active_fraction=float((gate>0).mean()),old_fraction=float((old_gate>0).mean()),
                     components=n-1,components_under4=sum(int(v<4) for v in stats[1:,4]))
            records.append(row);print('CONTOUR_ONLY',row,flush=True)
        write_json(dest/'verification.json',dict(parameters=cfg,oracle_errors=errors,source_sha256=file_hash(__file__),
            weight_sha256=file_hash(dest/'coefficients-padded.bin'),graph_unchanged=True,contour_only=True,
            used_gate_equal_visualized_gate=True,no_colour_acceptance=True))
    # Native-scale crops: no thumbnail shrinking that hides broken/extra lines.
    crops=[('val_stairs_090',(120,360,540,640),'草丛'),('val_stairs_090',(780,70,1200,350),'建筑'),
           ('val_plaza_090',(0,180,420,460),'树冠'),('val_plaza_210',(300,370,720,650),'人物')]
    board=Image.new('RGB',(2100,314*len(crops)),'#17202a');draw=ImageDraw.Draw(board)
    for row,(key,box,label) in enumerate(crops):
        rgb,_,old_gate=sample[key]
        oldmap=Image.fromarray(np.clip(old_gate*4,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST)
        imgs=[Image.fromarray(rgb),oldmap]+[Image.open(OUT/n/(key+'_gate.png')) for n in VARIANTS]
        for col,(im,title) in enumerate(zip(imgs,['原片 '+label,'旧：杂乱响应','C1 连贯结构','C2 减碎线','C3 再减弱线'])):
            board.paste(im.crop(box).convert('RGB'),(420*col,314*row+34));draw.text((420*col+5,314*row+4),title,font=font,fill='white')
    board.save(OUT/'native_contour_comparison.png')
    write_json(OUT/'diagnostics.json',dict(records=records,native_crops=crops,
        warning='Density/component counts are diagnostics, never substitute for contour correctness or user acceptance.'))
    shutil.copy2(__file__,OUT/Path(__file__).name)


if __name__=='__main__':main()
