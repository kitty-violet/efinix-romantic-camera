"""Fit small color corrections on training sources only, never validation."""
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .train_native_contours import ROOT,tensor,load_rgb
from .train_b1_distillation import to_rgb
from .stroke_color_input_net import StrokeColorInputNet
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def robust_affine(x,y,identity,ridge=.003):
    a=np.concatenate([x,np.ones((len(x),1))],1).astype(np.float64);b=y.astype(np.float64)
    coef=identity.copy();weights=np.ones(len(x))
    for _ in range(5):
        gram=a.T@(a*weights[:,None])/weights.sum();cross=a.T@(b*weights[:,None])/weights.sum()
        regularizer=np.eye(a.shape[1])*ridge
        coef=np.linalg.solve(gram+regularizer,cross+regularizer@identity)
        error=np.linalg.norm(a@coef-b,axis=1);weights=np.minimum(1,.06/np.maximum(error,1e-6))
    return coef.astype(np.float32)


@torch.no_grad()
def main():
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    old=ROOT/'native-stroke-color-texture-v4';out=ROOT/'seven-structured-fill-cool-v2';out.mkdir(exist_ok=False)
    data=json.loads((old/'data.json').read_text());state=torch.load(old/'generator_fp32.pt',map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model']);groups={}
    for i,r in enumerate(data['pairs']):groups.setdefault(r['source'],[]).append(i)
    before=[];targets=[];records=[]
    for source,indices in groups.items():
        for i in [indices[0],indices[len(indices)//2],indices[-1]]:
            row=data['pairs'][i];x=load_rgb(row['current']);t=load_rgb(row['current_target']);p=to_rgb(net(tensor(x)[None].cuda())[0])
            # Block averages reduce subpixel teacher geometry mismatch; this
            # does not smooth any displayed output or use validation targets.
            a=cv2.resize(p.astype(np.float32)/255,(160,90),interpolation=cv2.INTER_AREA)
            b=cv2.resize(t.astype(np.float32)/255,(160,90),interpolation=cv2.INTER_AREA)
            before.append(a.reshape(-1,3));targets.append(b.reshape(-1,3))
            records.append(dict(index=i,source=source,current=row['current'],target=row['current_target'],
                current_sha256=file_hash(row['current']),target_sha256=file_hash(row['current_target']),samples=a.shape[0]*a.shape[1]))
        print('COLOR_FIT_SOURCE',len(records),flush=True)
    x=np.concatenate(before);y=np.concatenate(targets)
    identity=np.concatenate([np.eye(3),np.zeros((1,3))]).astype(np.float64);rgb=robust_affine(x,y,identity)
    lx=cv2.cvtColor(x[None].astype(np.float32),cv2.COLOR_RGB2Lab)[0]/100
    ly=cv2.cvtColor(y[None].astype(np.float32),cv2.COLOR_RGB2Lab)[0]/100
    lab=robust_affine(lx,ly[:,1:],identity[:,1:])
    np.savez(out/'color_adapter.npz',rgb_affine=rgb,lab_chroma_affine=lab)
    pred=np.concatenate([x,np.ones((len(x),1))],1)@rgb
    write_json(out/'color_fit.json',dict(method='robust affine fit, 21 native720 images from all7 TRAIN sources; 8x8 block means',
        records=records,validation_used_for_fit=False,rgb_affine=rgb.tolist(),lab_chroma_affine=lab.tolist(),
        training_RGB_MAE_before=float(np.abs(x-y).mean()*255),training_RGB_MAE_after=float(np.abs(pred-y).mean()*255),
        source_weights_sha256=file_hash(old/'generator_fp32.pt'),source_code_sha256=file_hash(__file__)))
    print('COLOR_ADAPTER_FITTED',rgb.tolist(),lab.tolist(),flush=True)


if __name__=='__main__':main()
