"""Real continuous integer experiment, explicitly not an FPGA result."""
import json,time
import cv2,numpy as np
from PIL import Image
from .screen_integer_diffusion import diffuse,palette
from .train_luma_regions import ROOT,RUN
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_region_classes import encode,motion_map
from .render_wavelet_game import close_pipe,video_check
from .runtime import file_hash,write_json


def main():
    cv2.setNumThreads(2);out=ROOT/'runs'/RUN/'integer-diffusion-video-v1';out.mkdir(exist_ok=False)
    clip=next(c for c in json.loads((ROOT/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'] if c['id']=='stairs');assert file_hash(clip['clip'])==clip['clip_sha256']
    target=out/'stairs_diffusion_t12_n16_1080p.mp4';cap=cv2.VideoCapture(clip['clip']);encoder=encode(target);logs=[];previous=None;n=0;same=0;start=time.perf_counter()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pred=palette(diffuse(source,12,16));full=bilinear_2to3(pred);encoder.stdin.write(full.tobytes())
            if n in (90,120,210):Image.fromarray(pred).save(out/f'stairs_{n:03d}_native720.png');Image.fromarray(full).save(out/f'stairs_{n:03d}_preencode1080.png')
            if previous is not None:
                mx,my,mask=motion_map(previous[0],source);row=dict(frame=n)
                for name,a,b in [('source',previous[0],source),('output',previous[1],pred)]:
                    warped=cv2.remap(a.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);row[name]=float(np.abs(b.astype(np.float32)-warped).mean(2)[mask].mean())
                logs.append(row);same+=int(np.array_equal(previous[1],pred))
            previous=(source,pred);n+=1
            if n%30==0:print('INTEGER_DIFFUSION_VIDEO',n,round(time.perf_counter()-start,2),flush=True)
    finally:cap.release();close_pipe(encoder)
    assert n==360;check=video_check(target,n);motion={k:dict(mean=float(np.mean([v[k] for v in logs])),p95=float(np.percentile([v[k] for v in logs],95))) for k in ['source','output']}
    write_json(out/'motion.json',dict(records=logs,summary=motion,temporal_acceptance=False));write_json(out/'report.json',dict(status='INTEGER_ALGORITHM_VIDEO_NOT_ACCEPTED',video=check,path=str(target),source=clip['clip'],source_sha256=clip['clip_sha256'],frames=n,seconds=n/30,identical_adjacent_outputs=same,parameters=dict(threshold=12,steps=16,fraction_bits=4),pipeline='1080 integer area -> native720 integer nonlinear diffusion and palette -> integer bilinear1080',temporal_mixing=False,fpga=False,trainable_cnn=False,reference_only=True,motion=motion));print('INTEGER_DIFFUSION_VIDEO_DONE',motion,flush=True)


if __name__=='__main__':main()
