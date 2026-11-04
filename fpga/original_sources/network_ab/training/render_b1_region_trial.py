"""Render an explicitly experimental byte package, with real frame validation."""
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .fpga_video_resize import area_3to2,bilinear_2to3,verify
from .publish_moderate_contours import encode
from .render_wavelet_game import close_pipe,video_check
from .publish_region_classes import motion_map


def region_temporal_diagnostics(previous,current,mx,my,valid):
    """Separate solid-region colour flicker from transported contour drift.

    Overall RGB warp error conflates moving sharp edges and colour changes.
    These additional measures never modify the displayed pixels.
    """
    def edges(rgb):
        x=rgb.astype(np.float32);out=np.zeros(x.shape[:2],np.uint8)
        out[:,1:]|=(np.abs(np.diff(x,axis=1)).max(2)>8).astype(np.uint8)
        out[1:]|=(np.abs(np.diff(x,axis=0)).max(2)>8).astype(np.uint8)
        return out
    boundary=edges(current)
    prior=cv2.remap(edges(previous),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
    near=cv2.dilate(boundary|prior,np.ones((9,9),np.uint8)).astype(bool)
    interior=valid&~near;warped=cv2.remap(previous.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    delta=np.abs(current.astype(np.float32)-warped).mean(2)[interior]
    result=dict(interior_valid_pixels=int(interior.sum()),interior_fraction=float(interior.sum()/max(1,valid.sum())))
    result['interior']=dict(mean=float(delta.mean()),p95=float(np.percentile(delta,95)),
        over8=float((delta>8).mean()),over16=float((delta>16).mean())) if len(delta) else None
    distances=[]
    if boundary.any() and prior.any():
        for a,b in ((boundary,prior),(prior,boundary)):
            dt=cv2.distanceTransform(1-a,cv2.DIST_L2,5);distances.append(dt[(b>0)&valid])
    dist=np.concatenate(distances) if distances else np.empty(0)
    result['contours']=dict(mean_distance720=float(dist.mean()),p95_distance720=float(np.percentile(dist,95)),
        over2px=float((dist>2).mean()),over4px=float((dist>4).mean())) if len(dist) else None
    return result


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--scenes',nargs='+',default=['stairs','plaza'])
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=ap.parse_args()
    root=configure_workspace();set_reproducible(2026100525);contract_guard();cv2.setNumThreads(2);torch.set_num_threads(2)
    weights=a.package/'coefficients-padded.bin';review=json.loads((a.package/'review.json').read_text(encoding='utf-8'))
    assert review['integer'] and review['independent_oracle_error']==0
    assert all(r['export_node_error']==0 for r in review['records'])
    digest=file_hash(weights);assert digest==review['package_sha256']
    net=IntegerB1(runtime_from_image(weights.read_bytes()),device=a.device);a.out.mkdir(exist_ok=False)
    write_json(a.out/'resize.json',verify());clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    results=[]
    for clip in clips:
        if clip['id'] not in a.scenes:continue
        assert file_hash(clip['clip'])==clip['clip_sha256'];cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert 300<=total<=600 and cap.get(cv2.CAP_PROP_FPS)==30
        path=a.out/(clip['id']+'_integer1080.mp4');pipe=encode(path);n=0;previous=None;metrics=[];saved=[];start=time.perf_counter()
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                raw=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert raw.shape==(1080,1920,3);x=area_3to2(raw)
                q=net(torch.tensor(x.transpose(2,0,1)[None].copy(),device=a.device,dtype=torch.float32)-128)
                rgb=(q[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8);output=bilinear_2to3(rgb);pipe.stdin.write(output.tobytes())
                if previous is not None and n%5==0:
                    mx,my,valid=motion_map(previous[0],x);row=dict(frame=n)
                    for key,old,current in [('source',previous[0],x),('output',previous[1],rgb)]:
                        warped=cv2.remap(old.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        values=np.abs(current.astype(np.float32)-warped).mean(2)[valid]
                        row[key]=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)),over16=float((values>16).mean()))
                    row['region_diagnostics']=region_temporal_diagnostics(previous[1],rgb,mx,my,valid)
                    metrics.append(row)
                previous=(x,rgb)
                if n in (90,91,92,total-1):
                    Image.fromarray(output).save(a.out/f'{clip["id"]}_{n:03d}_integer1080.png')
                    if n==90:Image.fromarray(raw).save(a.out/f'{clip["id"]}_{n:03d}_source1080.png')
                    saved.append(n)
                n+=1
                if n%60==0:print('REGION_VIDEO',clip['id'],n,total,round(time.perf_counter()-start,1),flush=True)
        finally:cap.release();close_pipe(pipe)
        assert n==total and file_hash(weights)==digest;check=video_check(path,total)
        encoded=cv2.VideoCapture(str(path));errors=[]
        for frame in saved:
            encoded.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=encoded.read();assert ok
            ref=np.array(Image.open(a.out/f'{clip["id"]}_{frame:03d}_integer1080.png'));delta=np.abs(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.int16)-ref.astype(np.int16))
            errors.append(dict(frame=frame,mae=float(delta.mean()),p99=float(np.percentile(delta,99))))
        encoded.release();results.append(dict(id=clip['id'],video=check,source=clip['clip'],source_sha256=clip['clip_sha256'],motion=metrics,encoding=errors))
        write_json(a.out/'results.json',dict(package=str(weights),sha256=digest,results=results,
            chain='1080 RGB -> integer area720 -> reloaded B1 INT8/Q17 -> integer bilinear1080',
            postfilter=False,teacher_inference=False,actual_board=False,visual_accepted=False,inference_device=a.device))
    assert len(results)==len(a.scenes)
    links=[]
    for r in results:
        source=Path(os.path.relpath(r['source'],a.out)).as_posix();vid=r['id']+'_integer1080.mp4'
        links.append(f'<h2>{r["id"]} · {r["video"]["seconds"]:g}秒</h2><div class="pair"><section>原片<video controls preload="metadata" src="{source}"></video></section><section>B1真实整数包试验<video controls preload="metadata" src="{vid}"></video></section></div>')
    page='''<!doctype html><meta charset="utf-8"><title>B1颜色分区学习试验</title><style>body{margin:24px;background:#141b24;color:#eee;font:17px system-ui}video{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}section{min-width:0}p{max-width:1100px;line-height:1.6}</style><h1>B1颜色分区学习试验</h1><p>这是20 KiB实际系数重载后的软件整数模拟。没有额外后处理、教师参与或人物分割。按1080→720→1080处理，当前仍为画质审查版本，未宣称上板或风格验收通过。双击播放器可全屏查看。</p>'''+''.join(links)
    (a.out/'preview.html').write_text(page,encoding='utf-8');print('REGION_VIDEOS_COMPLETE',str(a.out),flush=True)


if __name__=='__main__':main()
