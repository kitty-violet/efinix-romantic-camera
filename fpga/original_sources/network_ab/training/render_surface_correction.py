"""Render source-guided surface correction with the user-accepted frozen weights."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .runtime import configure_workspace,file_hash,write_json
from .trainable_animegan import TrainableAnimeGAN
from .train_clear_style import tensor
from .refine_pretrained_style import pixels
from .smooth_surface_guard import correct_surface,CONFIG
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe,video_check


def main():
    root=configure_workspace();run=root/'runs/highlight-diagnosis-20261003';parent=root/'runs/clear-style-control-20261003';accepted=root/'runs/texture-recovery-20261003'
    torch.set_num_threads(4);cv2.setNumThreads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=accepted/'final/refined_style_fp32.pt';sha=file_hash(weights)
    assert sha=='c86b9b99e901c0ad6be7ce74bd35388a5028dc8c12228c579489025fa6b2f7a5'
    state=torch.load(weights,map_location='cpu',weights_only=True)
    net=TrainableAnimeGAN(root).cuda().eval();net.load_state_dict(state['generator'])
    clips=json.loads((parent/'video_clips.json').read_text(encoding='utf-8'))['clips'];out=run/'corrected_videos';out.mkdir(exist_ok=True);records=[]
    for row in clips:
        if row['id'] not in ['new_hallway','stairs']:continue
        src=Path(row['clip']);assert file_hash(src)==row['clip_sha256'];count=row['validation']['frames']
        path=out/f'{row["id"]}_corrected_1080p_12s.mp4'
        if path.exists():raise FileExistsError(path)
        cap=cv2.VideoCapture(str(src));pipe=encode_pipe(path,1920,1080,30);n=0;samples=[];times=[];stats=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);small=np.array(Image.fromarray(rgb).resize((1280,720),Image.Resampling.BICUBIC))
            start=time.perf_counter()
            with torch.no_grad():before=pixels(net(tensor(small)))
            result=correct_surface(small,before);times.append(time.perf_counter()-start)
            diff=np.abs(result.astype(float)-before);stats.append({'frame':n,'mean_rgb_change':float(diff.mean()),'fraction_unchanged':float((diff.max(2)==0).mean())})
            full=np.array(Image.fromarray(result).resize((1920,1080),Image.Resampling.LANCZOS));pipe.stdin.write(full.tobytes())
            if n in [0,90,120,180,196,210,240,270,359]:
                Image.fromarray(small).save(out/f'{row["id"]}_{n:03d}_source720.png')
                Image.fromarray(before).save(out/f'{row["id"]}_{n:03d}_before720.png')
                Image.fromarray(result).save(out/f'{row["id"]}_{n:03d}_native720.png')
                Image.fromarray(full).save(out/f'{row["id"]}_{n:03d}_preencode1080.png');samples.append(n)
            n+=1
            if n%90==0:print('CORRECTED',row['id'],n,'mean_seconds',np.mean(times),flush=True)
        cap.release();close_pipe(pipe);assert n==count
        records.append({'id':row['id'],'source':str(src),'source_sha256':row['clip_sha256'],'output':str(path),'validation':video_check(path,n),'samples':samples,'inference_calls':n,'mean_seconds':float(np.mean(times)),'weights_sha256':sha,'pixel_change':stats})
        write_json(out/'report.json',{'status':'COMPLETE_REQUIRES_REVIEW','records':records,'pipeline':'1080 -> 720 FP32 accepted network -> fixed source-guided surface guard -> 1080 H264 CRF16','weights_unchanged':True,'weights':str(weights),'correction_config':CONFIG,'source_sha256':{str(Path(__file__)):file_hash(__file__),str(Path(__file__).with_name('smooth_surface_guard.py')):file_hash(Path(__file__).with_name('smooth_surface_guard.py'))},'postprocess':'Spatial correction only; no exposure change, histogram adaptation, temporal filtering, semantic masks, or face restoration'})
    print('SURFACE_RENDER_COMPLETE',flush=True)


if __name__=='__main__':main()
