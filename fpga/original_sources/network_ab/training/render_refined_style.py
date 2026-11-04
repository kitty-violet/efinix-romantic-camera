"""Export real refined weights and render complete held-out 12/12/15 s clips."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .trainable_animegan import TrainableAnimeGAN
from .refine_pretrained_style import pixels
from .train_clear_style import tensor
from .runtime import configure_workspace, file_hash, write_json
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe, video_check


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True,type=Path);args=ap.parse_args()
    root=configure_workspace();run=root/'runs/texture-recovery-20261003';parent=root/'runs/clear-style-control-20261003'
    torch.set_num_threads(4);cv2.setNumThreads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    for p,sha in state['binding']['code'].items():
        if file_hash(p)!=sha:raise ValueError('Frozen source drift '+p)
    final=run/'final';final.mkdir(exist_ok=True)
    export=final/'refined_style_fp32.pt'
    if export.exists():raise FileExistsError(export)
    torch.save({'schema':'pretrained-animation-refined-fp32-v1','generator':state['ema'],'step':state['step'],
                'architecture':'TrainableAnimeGAN / AnimeGANv3 Hayao_36 decoder refinement',
                'binding':state['binding'],'config':state['config'],'checkpoint_sha256':file_hash(args.checkpoint),
                'scope':'Experimental animation teacher, not B1 or Dying Light weights'},export)
    loaded=torch.load(export,map_location='cpu',weights_only=True)
    net=TrainableAnimeGAN(root).cuda().eval();net.load_state_dict(loaded['generator'])
    weight_sha=file_hash(export)
    out=run/'refined_videos';out.mkdir(exist_ok=True);records=[]
    clips=json.loads((parent/'video_clips.json').read_text(encoding='utf-8'))['clips']
    for row in clips:
        src=Path(row['clip']);assert file_hash(src)==row['clip_sha256'];count=row['validation']['frames']
        path=out/f'{row["id"]}_refined_1080p_{count//30}s.mp4'
        if path.exists():raise FileExistsError(path)
        cap=cv2.VideoCapture(str(src));pipe=encode_pipe(path,1920,1080,30);n=0;times=[];previous=None;duplicates=0;stamps=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            small=np.array(Image.fromarray(rgb).resize((1280,720),Image.Resampling.BICUBIC))
            start=time.perf_counter()
            with torch.no_grad():result=pixels(net(tensor(small)))
            times.append(time.perf_counter()-start)
            full=np.array(Image.fromarray(result).resize((1920,1080),Image.Resampling.LANCZOS));data=full.tobytes()
            duplicates+=int(previous==data);previous=data;pipe.stdin.write(data)
            if n in (0,count//4,count//2,3*count//4,count-1):
                Image.fromarray(result).save(out/f'{row["id"]}_{n:03d}_native720.png')
                Image.fromarray(full).save(out/f'{row["id"]}_{n:03d}_preencode1080.png');stamps.append(n)
            n+=1
            if n%90==0:print('RENDER',row['id'],n,'mean_s',round(float(np.mean(times)),4),flush=True)
        cap.release();close_pipe(pipe);assert n==count
        records.append({'id':row['id'],'source':str(src),'source_sha256':row['clip_sha256'],'output':str(path),
                        'output_sha256':file_hash(path),'validation':video_check(path,n),'inference_calls':n,
                        'identical_adjacent_outputs':duplicates,'samples':stamps,'mean_seconds':float(np.mean(times)),'model_sha256':weight_sha})
        write_json(out/'report.json',{'status':'RENDERED_REQUIRES_VISUAL_REVIEW','records':records,'checkpoint':str(args.checkpoint),
          'pipeline':'1920x1080 -> PIL bicubic 1280x720 -> FP32 network -> PIL Lanczos 1920x1080',
          'postprocess':'Resize only. No source blend, face paste, temporal postfilter, frame duplication or interpolation.',
          'encoding':'H264 CRF16 yuv420p 30fps playback; not real-time inference'})
    write_json(final/'export.json',{'path':str(export),'sha256':weight_sha,'step':state['step'],'checkpoint':str(args.checkpoint),
                                 'source':state['binding'],'dtype':'FP32','requires_independent_reload_verification':True})
    print('RENDER_COMPLETE',flush=True)


if __name__=='__main__':main()
