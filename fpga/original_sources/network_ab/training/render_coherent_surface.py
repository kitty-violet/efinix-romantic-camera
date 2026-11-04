"""Render every frozen scene with one learned checkpoint and fixed 720p inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

from .coherent_surface_model import SurfaceAppearanceNet
from .render_temporal_native_detail import encode_pipe
from .runtime import configure_workspace,file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--clips',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure_workspace();torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['schema']!='coherent-surface-checkpoint-v1':raise ValueError('Wrong checkpoint')
    model_path=Path(__file__).with_name('coherent_surface_model.py')
    if file_hash(model_path)!=state['binding']['code'][str(model_path)]:raise ValueError('Model source drift')
    data=json.loads(args.clips.read_text(encoding='utf-8'))
    if data['status']!='SOURCE_VISUALLY_REVIEWED':raise ValueError('Unreviewed evaluation sources')
    if data['training_manifest_sha256']!=state['binding']['manifest_sha256']:raise ValueError('Split mismatch')
    net=SurfaceAppearanceNet(**state['config']['model']).cuda().eval()
    net.load_state_dict(state['ema'],strict=True)
    args.output.mkdir(parents=True,exist_ok=True)
    weight_path=args.output/'shared_style_fp32.pt'
    torch.save({'schema':'surface-appearance-fp32-weights-v1','config':state['config']['model'],
                'generator':state['ema'],'training_steps':state['step']+state['config'].get('prior_training_steps',0),
                'checkpoint_sha256':file_hash(args.checkpoint),'binding':state['binding']},weight_path)
    weights_sha=file_hash(weight_path);records=[]
    # One continuous preview, each scene stays at full 1920x1080 for five seconds.
    combined=args.output/f"all_scenes_1080p30_{len(data['clips'])*5}s.mp4"
    all_pipe=encode_pipe(combined,1920,1080,30)
    try:
        for row in data['clips']:
            src=Path(row['clip'])
            if file_hash(src)!=row['clip_sha256']:raise ValueError('Evaluation source drift')
            cap=cv2.VideoCapture(str(src))
            path=args.output/f"{row['id']}_styled_1080p.mp4"
            pipe=encode_pipe(path,1920,1080,30);count=0;seconds=[]
            previous=None;duplicates=0
            try:
                while True:
                    ok,bgr=cap.read()
                    if not ok:break
                    if bgr.shape!=(1080,1920,3):raise ValueError('Unexpected source size')
                    # Match the reviewed training masters' PIL bicubic reduction.
                    source_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    rgb=np.array(Image.fromarray(source_rgb).resize((1280,720),Image.Resampling.BICUBIC))
                    x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/128-1
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.no_grad():y=net(x)
                    torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
                    if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite output')
                    result=(y[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                    final=cv2.resize(result,(1920,1080),interpolation=cv2.INTER_LANCZOS4)
                    if previous is not None and np.array_equal(previous,final):duplicates+=1
                    previous=final
                    pipe.stdin.write(final.tobytes());all_pipe.stdin.write(final.tobytes())
                    if count in (0,75,149):
                        source_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                        panel=np.concatenate((source_rgb,final),axis=1)
                        Image.fromarray(panel).save(args.output/f"{row['id']}_{count:03d}_compare.jpg",quality=96)
                        Image.fromarray(final).save(args.output/f"{row['id']}_{count:03d}_styled.png")
                    count+=1
            finally:cap.release();pipe.stdin.close()
            error=pipe.stderr.read().decode('utf-8',errors='replace')
            if pipe.wait(timeout=120) or count!=150 or duplicates:raise RuntimeError(f'Invalid render {count}/{duplicates}: {error}')
            record={**row,'output':str(path),'output_sha256':file_hash(path),
                    'weights_sha256':weights_sha,'checkpoint_step':state['step'],
                    'total_training_steps':state['step']+state['config'].get('prior_training_steps',0),
                    'processing_resolution':[1280,720],'output_resolution':[1920,1080],
                    'frames_rendered':count,'identical_adjacent_frames':duplicates,
                    'median_gpu_inference_seconds':float(np.median(seconds)),
                    'input_resize':'PIL bicubic, matching training masters',
                    'postprocess':'Lanczos 720-to-1080 resize only; no per-scene fitting or temporal filter'}
            records.append(record)
            write_json(args.output/'render.json',{'schema':'shared-model-render-v1','status':'RENDERING',
                      'weights':str(weight_path),'weights_sha256':weights_sha,'clips':records})
            print('RENDERED',row['id'],count,flush=True)
    finally:all_pipe.stdin.close()
    errors=all_pipe.stderr.read().decode('utf-8',errors='replace')
    if all_pipe.wait(timeout=120):raise RuntimeError(errors)
    board=Image.new('RGB',(1920,len(records)*560),'#111');draw=ImageDraw.Draw(board)
    for i,row in enumerate(records):
        panel=Image.open(args.output/f"{row['id']}_075_compare.jpg").resize((1920,540),Image.Resampling.LANCZOS)
        board.paste(panel,(0,i*560+20));draw.text((8,i*560+3),f"{row['id']} | {row['split']} | SOURCE LEFT / SAME MODEL RIGHT",fill='white')
    board.save(args.output/'all_scenes_comparison.jpg',quality=95)
    write_json(args.output/'render.json',{'schema':'shared-model-render-v1','status':'VISUAL_REVIEW_REQUIRED',
          'weights':str(weight_path),'weights_sha256':weights_sha,'clips':records,
          'combined_video':str(combined),'combined_sha256':file_hash(combined),
          'limitation':'Software FP32 teacher evaluation; no claim of FPGA deployment or universal generalization.'})


if __name__=='__main__':main()
