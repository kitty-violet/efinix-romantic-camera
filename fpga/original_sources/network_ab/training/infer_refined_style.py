"""Independent FP32 single-image inference for the refined experimental teacher."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .runtime import configure_workspace, file_hash, write_json
from .trainable_animegan import TrainableAnimeGAN


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',required=True,type=Path)
    ap.add_argument('--image',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--compare',type=Path);args=ap.parse_args()
    root=configure_workspace();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(args.weights,map_location='cpu',weights_only=True)
    if state['schema']!='pretrained-animation-refined-fp32-v1':raise ValueError('Unexpected model schema')
    for p,sha in state['binding']['code'].items():
        if file_hash(p)!=sha:raise ValueError('Model code drift: '+p)
    net=TrainableAnimeGAN(root).cuda().eval();net.load_state_dict(state['generator'])
    im=Image.open(args.image).convert('RGB')
    if im.size!=(1280,720):im=im.resize((1280,720),Image.Resampling.BICUBIC)
    x=torch.from_numpy(np.array(im)).permute(2,0,1)[None].cuda().float()/128-1
    with torch.no_grad():y=net(x)
    a=(y[0].permute(1,2,0).cpu().numpy()*128+128).clip(0,255).astype(np.uint8)
    args.output.parent.mkdir(exist_ok=True,parents=True);Image.fromarray(a).save(args.output)
    report={'weights_sha256':file_hash(args.weights),'input':str(args.image),'output':str(args.output),'shape':list(a.shape)}
    if args.compare:
        b=np.array(Image.open(args.compare));d=np.abs(a.astype(float)-b);assert d.max()==0
        report.update(status='INDEPENDENT_RELOAD_PASS',pixel_max=float(d.max()),pixel_mae=float(d.mean()))
    write_json(args.output.with_suffix('.verification.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
