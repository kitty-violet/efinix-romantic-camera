"""Explicit contour barrier, surface, boundary and palette controls in B1."""
import copy
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible
from .b1_trained_image import Parameters,encode,decode,runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .fpga_video_resize import bilinear_2to3

ROOT=Path(r'E:\fpga\model-training-workspace\runs');BASE=ROOT/'b1-contour-gated-20261005-v1'
OUT=ROOT/'b1-contour-gated-20261005-v2'
VARIANTS={'structured':dict(edge=8,sharp=1,mix=.875),
          'crisp':dict(edge=8,sharp=2,mix=.875),
          'sparse':dict(edge=10,sharp=2,mix=.875)}


def modify(params,cfg):
    p=copy.deepcopy(params)
    old=p['down2.pw'];b=list(old.bias)
    for c in range(24,32):b[c]=-8*cfg['edge']
    p['down2.pw']=Parameters(old.weights,tuple(b),old.multiplier,old.relu)
    # Enhance only coarse luminance detail remaining AFTER contour-gated merging.
    old=p['up1_low_dw'];w=list(old.weights);b=list(old.bias);m=list(old.multiplier)
    lap=(0,-2,0,-2,8,-2,0,-2,0)
    w[3]=lap;w[6]=tuple(-v for v in lap);b[3]=b[6]=-16;m[3]=m[6]=16384
    p['up1_low_dw']=Parameters(tuple(w),tuple(b),tuple(m),True)
    old=p['up1_low_pw'];w=[list(a) for a in old.weights]
    for c,sign in [(0,1),(3,1),(4,-1)]:
        w[c][3]=cfg['sharp']*sign;w[c][6]=-cfg['sharp']*sign
    p['up1_low_pw']=Parameters(tuple(tuple(a) for a in w),old.bias,old.multiplier,True)
    # Controlled small within-block slope, using reserved colour head channels.
    levels=np.array([6,3,3]);phase=(-64)%levels;phase[0]=0
    old=p['up2_low_pw'];w=list(old.weights);b=list(old.bias);m=list(old.multiplier)
    for c in range(3):
        w[3+c]=w[c];b[3+c]=b[c];m[3+c]=int(m[c]*levels[c]);assert m[3+c]<131072
    p['up2_low_pw']=Parameters(tuple(w),tuple(b),tuple(m),True)
    old=p['up2_high_dw'];w=list(old.weights)
    for c in range(3,6):w[c]=w[0]
    p['up2_high_dw']=Parameters(tuple(w),old.bias,old.multiplier,True)
    # Coherent cold, saturated colours; one RGB matrix shared over all scenes.
    con,sat,mix=1.03125,1.125,cfg['mix']
    matrix=np.array([[con,sat,-sat],[con,0,sat],[con,-sat,-sat]])
    w=np.zeros((3,8),int);bias=[]
    for c in range(3):
        w[c,:3]=np.rint(8*matrix[c]*levels*mix).astype(int)
        w[c,3:6]=np.rint(8*matrix[c]*(1-mix)).astype(int)
        bias.append(int(np.rint(-np.sum(w[c,:3]*(64+phase)/levels)-np.sum(w[c,3:6]*(64+phase))+4*[-1,0,2][c])))
    p['rgb_head']=Parameters(tuple(tuple(int(v) for v in a) for a in w),tuple(bias),tuple([32768]*3),False)
    if cfg.get('ink',0):
        # Dark-side contour accent from the ALREADY MERGED coarse luminance,
        # not a high-frequency edge overlay extracted from the raw photograph.
        old=p['up1_low_pw'];w=[list(a) for a in old.weights];w[5][6]=2
        p['up1_low_pw']=Parameters(tuple(tuple(a) for a in w),old.bias,old.multiplier,True)
        old=p['up1_high_dw'];w=list(old.weights);w[5]=w[0]
        p['up1_high_dw']=Parameters(tuple(w),old.bias,old.multiplier,True)
        old=p['up2_low_pw'];w=list(old.weights);b=list(old.bias);m=list(old.multiplier)
        row=[0]*16;row[5]=2;w[6]=tuple(row);b[6]=0;m[6]=65536
        p['up2_low_pw']=Parameters(tuple(w),tuple(b),tuple(m),True)
        old=p['up2_high_dw'];w=list(old.weights);w[6]=w[0]
        p['up2_high_dw']=Parameters(tuple(w),old.bias,old.multiplier,True)
        old=p['rgb_head'];w=[list(a) for a in old.weights]
        for c in range(3):w[c][6]=-4*cfg['ink']
        p['rgb_head']=Parameters(tuple(tuple(a) for a in w),old.bias,old.multiplier,False)
    return p


@torch.inference_mode()
def main(ink=False):
    global OUT,VARIANTS
    if ink:
        OUT=ROOT/'b1-contour-gated-20261005-v3'
        VARIANTS={name:dict(edge=8,sharp=2,mix=.875,ink=value) for name,value in [('base',0),('outline',1),('outline_strong',2)]}
    configure_workspace();set_reproducible(2026100504);contract_guard();OUT.mkdir(exist_ok=False)
    params,residual=decode((BASE/'blocks/coefficients-padded.bin').read_bytes())
    cases=json.loads((ROOT/'b1-deploy-preview-20261005-v1/data.json').read_text(encoding='utf-8'))['validation']
    board=Image.new('RGB',(2100,266*len(cases)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for i,r in enumerate(cases):
        for j,(path,label) in enumerate([(r['source'],'原片'),(r['target'],'冻结目标')]):
            board.paste(Image.open(path).resize((420,236)),(420*j,266*i+30));draw.text((420*j+4,266*i+3),label,font=font,fill='white')
    for column,(name,cfg) in enumerate(VARIANTS.items(),2):
        dest=OUT/name;dest.mkdir();p=modify(params,cfg);raw=encode(p,residual);padded=raw+bytes(20480-len(raw))
        (dest/'coefficients.bin').write_bytes(raw);(dest/'coefficients-padded.bin').write_bytes(padded)
        (dest/'load_words.hex').write_text(''.join(f'{int.from_bytes(padded[i:i+8],"little"):016x}\n' for i in range(0,len(padded),8)))
        net=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
        checks=[]
        for i,r in enumerate(cases):
            rgb=np.array(Image.open(r['source']));x=torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda')
            nodes,_=net(x,True);out=(nodes['rgb_head'][0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
            Image.fromarray(out).save(dest/(r['id']+'_720.png'));Image.fromarray(bilinear_2to3(out)).save(dest/(r['id']+'_1080.png'))
            contour=nodes['down2.pw'][0,24].cpu().numpy()
            Image.fromarray(np.clip(contour*4,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).save(dest/(r['id']+'_contour.png'))
            board.paste(Image.fromarray(out).resize((420,236)),(420*column,266*i+30));draw.text((420*column+4,266*i+3),'轮廓约束：'+name,font=font,fill='white')
            checks.append(dict(id=r['id'],mae_target=float(np.abs(out.astype(float)-np.array(Image.open(r['target']))).mean()),contour_fraction=float((contour>0).mean())))
        write_json(dest/'verification.json',dict(status='CONTOUR_GATED_B1_VISUAL_CANDIDATE',config=cfg,checks=checks,
            sha256=file_hash(dest/'coefficients-padded.bin'),designed_surface=True,graph_unchanged=True,
            contour_guided=True,extra_postprocess=False,actual_board=False,visual_accepted=False))
        print('CONTOUR_CONTROL',name,checks,flush=True)
    board.save(OUT/'comparison.jpg',quality=96);shutil.copy2(__file__,OUT/Path(__file__).name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ink',action='store_true');args=parser.parse_args();main(args.ink)
