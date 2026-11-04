"""Replay the complete experimental package on a real image (not weights-only)."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from .stroke_color_input_net import StrokeColorInputNet
from .train_native_contours import tensor,load_rgb
from .train_b1_distillation import to_rgb
from .coarse_contour_targets import target_parameters
from .contour_guided_fill import fill
from .adjust_structured_color import adapt_lab
from .fpga_video_resize import area_3to2,bilinear_2to3
from .engine import set_reproducible
from .runtime import configure_workspace,contract_guard,file_hash


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--level',choices=[f'g{i}' for i in range(1,8)],default='g4');args=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=json.loads((args.package/'pipeline.json').read_text(encoding='utf-8'));path=args.package/'student_RGB_fp32.pt'
    assert file_hash(path)==cfg['source_weight_sha256'];state=torch.load(path,map_location='cpu',weights_only=True)
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model'])
    a=load_rgb(args.input);up=a.shape[:2]==(1080,1920)
    if up:a=area_3to2(a)
    assert a.shape[:2]==(720,1280),'Expected native720 or 1920x1080 input; no implicit resizing.'
    pred=to_rgb(net(tensor(a)[None].cuda())[0]);params=dict(cfg['levels'][args.level]);params['sigma']+=cfg['contour_sigma_increment']
    edge=target_parameters(pred,params);result=fill(pred,edge,cfg['strength'],cfg['iterations'])
    result=adapt_lab(result,np.array(cfg['color_adapter_coefficients'],np.float32),1.15)
    if up:result=bilinear_2to3(result)
    assert not args.output.exists(),'Refusing to overwrite an existing output.'
    Image.fromarray(result).save(args.output);print('COMPLETE_PIPELINE_REPLAY',args.output,flush=True)


if __name__=='__main__':main()
