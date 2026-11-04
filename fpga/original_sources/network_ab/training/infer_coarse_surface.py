"""Reproduce the complete v3 experiment; refuse changed code or assets."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from .coarse_surface_fill import process, LEVELS, COLOR
from .stroke_color_input_net import StrokeColorInputNet
from .train_native_contours import tensor, load_rgb
from .train_b1_distillation import to_rgb
from .fpga_video_resize import area_3to2, bilinear_2to3
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--level',choices=list(LEVELS),default='g4')
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100487)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=json.loads((args.package/'pipeline.json').read_text(encoding='utf-8'))
    assert cfg['levels']==LEVELS and cfg['color']==COLOR,'Use the frozen source/config version.'
    assert all(file_hash(k)==v for k,v in cfg['code'].items()),'Source changed since publishing.'
    weight=args.package/'student_RGB_fp32.pt';adapter_file=args.package/'color_adapter.npz'
    assert file_hash(weight)==cfg['weight_sha256'] and file_hash(adapter_file)==cfg['adapter_sha256']
    state=torch.load(weight,map_location='cpu',weights_only=True)
    assert all(file_hash(k)==v for k,v in state['binding']['code'].items())
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model'])
    adapter=np.load(adapter_file)['lab_chroma_affine']
    def infer(a):
        up=a.shape[:2]==(1080,1920)
        if up:a=area_3to2(a)
        assert a.shape[:2]==(720,1280),'Only native720 or full1080 input is supported.'
        rgb=to_rgb(net(tensor(a)[None].cuda())[0])
        result,e=process(rgb,adapter,args.level)
        return bilinear_2to3(result) if up else result
    if args.verify:
        records=[]
        for row in json.loads((args.package/'data.json').read_text())['validation']:
            result=infer(load_rgb(args.package/(row['id']+'_source.png')))
            expected=load_rgb(args.package/(row['id']+'_'+args.level+'_color.png'))
            error=int(np.abs(result.astype(np.int16)-expected.astype(np.int16)).max())
            assert error==0
            records.append(dict(id=row['id'],max_pixel_error=error))
        write_json(args.package/'independent_replay.json',dict(level=args.level,records=records,
                   precision='FP32',code_and_weight_hashes_match=True))
        print('INDEPENDENT_COMPLETE_PIPELINE_EXACT',len(records),flush=True)
    else:
        assert args.input and args.output,'Provide --input and --output.'
        assert not args.output.exists(),'Refusing to overwrite output.'
        Image.fromarray(infer(load_rgb(args.input))).save(args.output)
        print('COMPLETE_COARSE_SURFACE_INFERENCE',args.output,flush=True)


if __name__=='__main__':main()
