"""Frozen 12/12/15-second source clips and identical-frame teacher/student renders."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from .runtime import configure_workspace, write_json, file_hash
from .media import decode_frames, probe_video
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe, video_check


def prepare(root, run):
    dst = run/'video_sources'; dst.mkdir(exist_ok=True)
    raw = next((root/'data/incoming/content').glob('*16*'))
    specs = [('new_hallway','VID_20260925_065914.mp4',0,360,'new_source_holdout'),
             ('stairs','VID_20260925_113231.mp4',30,360,'previously_seen_holdout'),
             ('plaza','VID_20260925_101046.mp4',210,450,'previously_seen_validation')]
    rows=[]
    for name, filename, start, count, role in specs:
        source=raw/filename; target=dst/(name+'_source_1080p.mp4')
        if target.exists(): raise FileExistsError(target)
        pipe=encode_pipe(target,1920,1080,30); frames=[]
        for ix, pts, arr in decode_frames(source,indices=list(range(start,start+count)),hlg_exposure_stops=-1):
            if arr.shape!=(1080,1920,3): raise ValueError('Unexpected source orientation/dimensions')
            pipe.stdin.write(arr.tobytes());frames.append({'index':ix,'pts':pts})
            if len(frames) in (1,count//2,count): Image.fromarray(arr).save(dst/f'{name}_{len(frames)-1:03d}_decoded.png')
        close_pipe(pipe)
        if len(frames)!=count:raise RuntimeError('Truncated clip')
        rows.append({'id':name,'source':str(source),'source_sha256':file_hash(source),'role':role,
                     'clip':str(target),'clip_sha256':file_hash(target),'frames':frames,'validation':video_check(target,count),
                     'metadata':probe_video(source),'note':'Every selected source frame once, 30fps playback; original PTS preserved in manifest'})
        print('PREPARED',name,count,count/30,'seconds',flush=True)
        write_json(run/'video_clips.json',{'status':'FROZEN_SOURCE_CLIPS','clips':rows})


def render(root, run, model, checkpoint):
    clips=json.loads((run/'video_clips.json').read_text(encoding='utf-8'))['clips']
    out=run/(model+'_videos');out.mkdir(exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if model in ('teacher','guarded'):
        from .animegan_teacher import AnimeTeacher,MODEL_SHA
        net=AnimeTeacher(root);weight_sha=MODEL_SHA
        if model=='teacher':infer=net
        else:
            from .guard_anime_teacher import guard_teacher
            infer=lambda rgb:guard_teacher(rgb,net(rgb))
    else:
        from .coherent_surface_model import SurfaceAppearanceNet
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=SurfaceAppearanceNet(**state['config']['model']).cuda().eval()
        net.load_state_dict(state['ema']);weight_sha=file_hash(checkpoint)
        def infer(rgb):
            x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda().float()/128-1
            with torch.no_grad():y=net(x)
            return (y[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
    records=[]
    for row in clips:
        source=Path(row['clip']);assert file_hash(source)==row['clip_sha256']
        count=row['validation']['frames'];path=out/(row['id']+f'_{model}_1080p_{count//30}s.mp4')
        cap=cv2.VideoCapture(str(source));pipe=encode_pipe(path,1920,1080,30)
        previous=None;duplicates=0;n=0;times=[];stamps=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            small=np.array(Image.fromarray(source_rgb).resize((1280,720),Image.Resampling.BICUBIC))
            start=time.perf_counter();result=infer(small);times.append(time.perf_counter()-start)
            final=np.array(Image.fromarray(result).resize((1920,1080),Image.Resampling.LANCZOS))
            data=final.tobytes();duplicates+=int(previous==data);previous=data;pipe.stdin.write(data)
            if n in (0,count//4,count//2,3*count//4,count-1):
                Image.fromarray(result).save(out/f'{row["id"]}_{n:03d}_native720.png')
                Image.fromarray(final).save(out/f'{row["id"]}_{n:03d}_preencode1080.png')
                stamps.append(n)
            n+=1
            if n%90==0:print(model,row['id'],n,'mean_seconds',round(float(np.mean(times)),4),flush=True)
        cap.release();close_pipe(pipe);assert n==count
        records.append({'id':row['id'],'model':model,'source':str(source),'source_sha256':row['clip_sha256'],
                        'output':str(path),'validation':video_check(path,n),'inference_calls':n,
                        'identical_adjacent_outputs':duplicates,'mean_seconds':float(np.mean(times)),
                        'samples':stamps,'model_sha256':weight_sha})
        write_json(out/'report.json',{'status':'RENDERED_REQUIRES_VISUAL_REVIEW','records':records,
           'pipeline':'1920x1080 -> PIL bicubic 1280x720 -> model -> PIL Lanczos 1920x1080',
           'postprocess':('training-only source-edge guidance suppresses smooth-region false contours; no temporal blending or face paste' if model=='guarded' else 'fixed resize only; no temporal blending, no face paste, no frame interpolation'),
           'encoding':'H264 CRF16 yuv420p; playback 30fps is not inference FPS'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','render']);ap.add_argument('--model',choices=['teacher','guarded','student'],default='teacher');ap.add_argument('--checkpoint',type=Path)
    args=ap.parse_args();root=configure_workspace();run=root/'runs/clear-style-control-20261003'
    if args.action=='prepare':prepare(root,run)
    else:render(root,run,args.model,args.checkpoint)


if __name__=='__main__':main()
