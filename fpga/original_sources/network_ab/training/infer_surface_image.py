"""Run the exported FP32 weights on one SDR RGB image, without downloads."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from coherent_surface_model import SurfaceAppearanceNet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path(__file__).with_name('shared_style_fp32.pt'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    state=torch.load(a.weights,map_location='cpu',weights_only=True)
    if state['schema']!='surface-appearance-fp32-weights-v1':
        raise ValueError('Unexpected weight schema')
    net=SurfaceAppearanceNet(**state['config']).to(a.device).eval()
    net.load_state_dict(state['generator'],strict=True)
    im=Image.open(a.input).convert('RGB')
    if im.size!=(1920,1080):raise ValueError('Expected a 1920x1080 SDR RGB image')
    rgb=np.array(im.resize((1280,720),Image.Resampling.BICUBIC))
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(a.device).float()/128-1
    with torch.no_grad():y=net(x)
    if not torch.isfinite(y).all():raise ValueError('Nonfinite prediction')
    rgb=(y[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
    result=cv2.resize(rgb,(1920,1080),interpolation=cv2.INTER_LANCZOS4)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(result).save(a.output)
    print(a.output)


if __name__=='__main__':main()
