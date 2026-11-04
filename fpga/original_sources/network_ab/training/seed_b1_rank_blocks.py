"""A nonlinear rank/colour-block initialization encoded entirely in B1.

Each residual uses comparisons implemented by DW/PW/ReLU to compute a
three-sample median. It chooses local colours instead of averaging across
boundaries. This is constructed initialization, not evidence of learning.
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .b1_trained_image import Parameters, encode, decode, runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .engineer_b1_blocks import oracle_records
from .train_b1_structural_gate import CROPS
from .train_b1_deep_contour import OUT as LABELS
from network_ab.contract_reference import execute_integer

OUT=LABELS.parent/'b1-rank-blocks-20261005-v1'


def build(alpha=.8, width=16., ink=.3):
    _,plan=contract_guard();arrays={};rows={r['id']:r for r in plan['nodes']}
    for r in plan['nodes']:
        if not r['kernel']:continue
        ci=1 if r['kind']=='dw3x3' else r['input_channels']
        arrays[r['id']]=[np.zeros((r['channels'],ci,r['kernel'],r['kernel']),np.float64),
                          np.zeros(r['channels'],np.float64)]
    def identity(name):
        w,b=arrays[name]
        for c in range(len(w)):w[c,c if w.shape[1]>1 else 0,w.shape[2]//2,w.shape[3]//2]=1
    w,b=arrays['stem']
    for c in range(3):w[c,c,1,1]=.5;b[c]=64
    gaussian=np.array([[1,2,1],[2,4,2],[1,2,1]],np.float64)/16
    w,b=arrays['down1.dw'];w[:3,0]=gaussian
    # Positive Y/Co/Cg; copying all lanes lets depthwise comparisons operate on
    # different neighbours without adding a gather or a new hardware operator.
    matrix=np.array([[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]])
    w,b=arrays['down1.pw']
    for c in range(16):w[c,:3,0,0]=matrix[c%3];b[c]=64 if c%3 else 0
    w,b=arrays['down2.dw'];w[:,0]=gaussian
    w,b=arrays['down2.pw']
    for c in range(32):w[c,c%3,0,0]=1
    for stage,(dy,dx) in enumerate([(0,1),(1,0),(1,1)],1):
        prefix=f'res{stage}_conv'
        w,b=arrays[prefix+'1.dw']
        for c in range(3):
            w[c,0,1-dy,1-dx]=1                 # a
            w[3+c,0,1,1]=1                    # b
            w[6+c,0,1+dy,1+dx]=1              # c
            w[9+c,0,1-dy,1-dx]=1;w[9+c,0,1,1]=-1 # relu(a-b)
        if stage==3:
            for slot,(yy,xx) in zip((15,18,21,24),((0,1),(2,1),(1,0),(1,2))):
                w[slot,0,yy,xx]=1;w[slot,0,1,1]=-1
        w,b=arrays[prefix+'1.pw']
        for c in range(3):
            w[c,c,0,0]=1
            w[3+c,9+c,0,0]=1
            w[6+c,6+c,0,0]=1;w[6+c,c,0,0]=-1;w[6+c,9+c,0,0]=1
            w[9+c,6+c,0,0]=1;w[9+c,3+c,0,0]=-1;w[9+c,9+c,0,0]=-1
            w[12+c,3+c,0,0]=1
        if stage==3:
            w[15,[15,18,21,24],0,0]=1;b[15]=-5
        identity(prefix+'2.dw')
        w,b=arrays[prefix+'2.pw']
        for j in range(32):
            c=j%3
            for slot,coefficient in [(c,1),(3+c,-1),(6+c,1),(9+c,-1),(12+c,-1)]:
                w[j,slot,0,0]=coefficient
            if stage==3 and c==0:w[j,15,0,0]=-ink
    identity('up1_low_dw')
    w,b=arrays['up1_low_pw']
    for c in range(16):w[c,c%3,0,0]=1
    # Interpolate the selected surfaces before the clipped colour ramps. The
    # nonlinear ramps then shorten transitions rather than blurring them again.
    w,b=arrays['up1_high_dw'];w[:,0]=gaussian
    identity('up2_low_dw')
    w,b=arrays['up2_low_pw']
    for c,threshold in enumerate((16,48,80,112)):
        w[c,0,0,0]=127/width;b[c]=-(threshold-width/2)*127/width
    w[4,0,0,0]=1;w[5,1,0,0]=1;w[6,2,0,0]=1
    identity('up2_high_dw')
    w,b=arrays['rgb_head']
    con,sat=1.,1.12
    for c in range(3):
        w[c,:4,0,0]=2*con*alpha*32/127
        w[c,4,0,0]=2*con*(1-alpha)
    for c,(co,cg) in enumerate(((1,-1),(0,1),(-1,-1))):
        w[c,5,0,0]=2*sat*co;w[c,6,0,0]=2*sat*cg
        b[c]=-128-128*sat*(co+cg)+(-2,0,2)[c]
    params={}
    for name,(w,b) in arrays.items():
        scale=np.maximum(np.ceil(np.maximum(np.abs(w).reshape(len(w),-1).max(1),1e-5)*131072/112),1)/131072
        # Prefer exact powers for the median comparisons / unchanged colour
        # carrier. All of those coefficients are multiples of 1/16.
        if name!='up2_low_pw' and name!='rgb_head' and not name=='res3_conv2.pw':scale[:]=1/32
        m=np.rint(scale*131072).astype(np.int64)
        wi=np.sign(w)*np.floor(np.abs(w)/scale[:,None,None,None]+.5)
        bi=np.sign(b)*np.floor(np.abs(b)/scale+.5)
        assert abs(wi).max()<=127 and abs(bi).max()<2**31 and 0<m.min()<=m.max()<131072
        flat=wi.transpose(0,2,3,1).reshape(len(w),-1).astype(int).tolist()
        params[name]=Parameters(tuple(map(tuple,flat)),tuple(bi.astype(int).tolist()),tuple(m.tolist()),rows[name]['activation']=='relu')
    residual={f'res{i}_add':dict(main_scaled=True,multiplier=131071,post_multiplier=131071) for i in range(1,4)}
    return params,residual


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100518);contract,plan=contract_guard();OUT.mkdir(exist_ok=False)
    # Exact median identity, independent of images/training scores.
    a,b,c=np.random.default_rng(13).integers(0,128,(3,100000))
    d=np.maximum(a-b,0)
    calculated=a-d+np.maximum(c-a+d,0)-np.maximum(c-b-d,0)
    assert np.array_equal(calculated,np.median(np.stack([a,b,c]),axis=0))
    data=json.loads((LABELS/'data.json').read_text(encoding='utf-8'));images={};records={}
    for key,alpha,width,ink in [('gentle',.65,20.,.2),('blocks',.9,16.,.3)]:
        folder=OUT/key;folder.mkdir();params,residual=build(alpha,width,ink)
        raw=encode(params,residual);raw+=bytes(20480-len(raw));(folder/'coefficients-padded.bin').write_bytes(raw)
        net=IntegerB1(runtime_from_image((folder/'coefficients-padded.bin').read_bytes()))
        op,rp=oracle_records(params);sample=np.random.default_rng(72).integers(-128,128,(12,12,3),dtype=np.int16)
        ref=execute_integer(contract,sample.tolist(),op,rp)
        nodes,_=net(torch.tensor(sample.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        error=max(int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][n])).max()) for n,v in nodes.items());assert error==0
        records[key]=dict(alpha=alpha,width=width,ink=ink,oracle_error=error,sha256=file_hash(folder/'coefficients-padded.bin'))
        for r in data:
            if r['id'] not in {c[0] for c in CROPS}:continue
            x=np.array(Image.open(r['source']).convert('RGB'))
            q=net(torch.tensor(x.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128)
            rgb=Image.fromarray((q[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8))
            images.setdefault(r['id'],dict(source=Image.fromarray(x),target=Image.open(r['target']).convert('RGB')))[key]=rgb
            rgb.save(folder/(r['id']+'_raw720.png'))
    board=Image.new('RGB',(420*4,314*4),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for i,(key,box,label) in enumerate(CROPS):
        for j,(kind,title) in enumerate([('source','原片'),('target','冻结目标'),('gentle','真实B1：中值归并初值 / 轻'),('blocks','真实B1：中值归并初值 / 强')]):
            board.paste(images[key][kind].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),title,font=font,fill='white')
    board.save(OUT/'actual_initializations.png')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    write_json(OUT/'verification.json',dict(constructed=True,trained=False,median_scalar_cases=100000,
        variants=records,unchanged_B1=True,postprocessing=False,actual_board=False,visual_accepted=False))
    print('RANK_BLOCKS_COMPLETE',records,flush=True)


if __name__=='__main__':main()
