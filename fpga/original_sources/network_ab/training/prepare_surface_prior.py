"""Fit one monotone tone prior to TRAIN sources only; save a held-out preview.

This prior regularizes a previously game-trained teacher. It is not fitted on test
videos and will not be part of the exported network's inference path.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from . import coherent_shared_train as base
from .coherent_shared_model import SharedAppearanceNet
from .coherent_surface_prior import repair_target
from .runtime import configure_workspace,file_hash,write_json


def isotonic(values,weights):
    blocks=[]
    for i,(v,w) in enumerate(zip(values,weights)):
        blocks.append([i,i,float(v)*w,float(w)])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    out=np.empty(len(values))
    for start,end,total,weight in blocks:out[start:end+1]=total/weight
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();configure_workspace();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=json.loads(a.manifest.read_text());state=torch.load(a.weights,map_location='cpu',weights_only=True)
    if state['binding']['manifest_sha256']!=file_hash(a.manifest):raise ValueError('Training split mismatch')
    if (a.output/'prior.json').exists():raise FileExistsError('Prior already frozen')
    a.output.mkdir(exist_ok=True,parents=True)
    net=SharedAppearanceNet(**state['config']).cuda().eval();net.load_state_dict(state['generator'])
    groups={}
    for row in data['content']:
        if row['split']=='train':groups.setdefault(row['source'],[]).append(row)
    inputs=[];outputs=[];used=[]
    rng=np.random.default_rng(20261008)
    with torch.no_grad():
        for source,rows in sorted(groups.items()):
            indices=np.linspace(0,len(rows)-1,min(4,len(rows))).round().astype(int)
            for index in indices:
                row=rows[index]
                if file_hash(row['path'])!=row['sha256']:raise ValueError('Source image drift')
                image=np.array(Image.open(row['path']).convert('RGB').resize((640,360),Image.Resampling.BICUBIC))
                out=net(base.tensor(image))[0].permute(1,2,0).cpu().numpy()*128+128
                pick=rng.choice(640*360,8000,replace=False)
                inputs.append(image.reshape(-1,3)[pick]/255.)
                outputs.append(out.reshape(-1,3)[pick]/255.)
                used.append({'source':source,'path':row['path'],'sha256':row['sha256'],'split':'train'})
    x,y=np.concatenate(inputs),np.concatenate(outputs);w=np.array([.2126,.7152,.0722])
    lx,ly=x@w,y@w;bins=np.clip(np.rint(lx*32).astype(int),0,32)
    counts=np.bincount(bins,minlength=33)
    values=np.array([np.median(ly[bins==i]) if counts[i] else np.nan for i in range(33)])
    have=np.isfinite(values);values=np.interp(np.arange(33),np.flatnonzero(have),values[have])
    values=isotonic(values,np.maximum(counts,1))
    # Gentle positive slopes prevent reversal; bounded slopes prevent excessive local contrast.
    values[0]=max(0,min(values[0],.015))
    for i in range(1,33):values[i]=np.clip(values[i],values[i-1]+.002,values[i-1]+.05)
    values=np.clip(values,0,1)
    mapped=np.interp(lx,np.linspace(0,1,33),values)
    design=np.stack((x[:,0]-lx,x[:,2]-lx,mapped),1)
    expected=y-ly[:,None]
    matrix=np.linalg.solve(design.T@design+np.eye(3)*.05,design.T@expected)
    matrix-= (matrix@w)[:,None]
    prior={'schema':'shared-tone-surface-prior-v1','luma_knots':values.tolist(),
       'chroma_matrix':matrix.tolist(),'used_training_frames':used,'training_sources':len(groups),
       'weights_sha256':file_hash(a.weights),'manifest_sha256':file_hash(a.manifest),
       'source_scope':'Training split only; one shared mapping; never fitted on validation/test.',
       'prior_note':'Tone inherited from game-trained teacher; surface simplification is explicit prior, not newly learned game texture.'}
    write_json(a.output/'prior.json',prior)
    validation={}
    for row in data['content']:
        if row['split']=='val':validation.setdefault(row['source'],[]).append(row)
    with torch.no_grad():
        for source,rows in sorted(validation.items()):
            row=rows[len(rows)//2];im=np.array(Image.open(row['path']).convert('RGB'));x=base.tensor(im);old=net(x)
            variants=[x,old]+[repair_target(x,old,prior,m) for m in ('light','texture','combined')]
            board=Image.new('RGB',(1280,380*len(variants)),'#151515');draw=ImageDraw.Draw(board)
            for i,(label,out) in enumerate(zip(('SOURCE','PREVIOUS 4250','LIGHT TARGET','TEXTURE TARGET','COMBINED TARGET'),variants)):
                pixels=(out[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                board.paste(Image.fromarray(pixels).resize((640,360)),(0,i*380+20))
                # Full-resolution center crop reveals whether contours survive fine-detail removal.
                board.paste(Image.fromarray(pixels).crop((320,280,960,640)),(640,i*380+20))
                draw.text((4,i*380+3),label+' / TRAINING TARGET PREVIEW, NOT NEW MODEL',fill='white')
            board.save(a.output/(Path(source).stem+'_targets.jpg'),quality=96)
    print(json.dumps({'status':'PRIOR_FROZEN','training_sources':len(groups),'frames':len(used),
       'white_luma':values[-1],'prior_sha256':file_hash(a.output/'prior.json')}),flush=True)


if __name__=='__main__':main()
