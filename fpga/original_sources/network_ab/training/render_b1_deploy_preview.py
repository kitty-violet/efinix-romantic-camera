"""Actual package bytes -> integer B1 -> project integer 1080p scaler videos."""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .train_b1_deploy_preview import OUT, ROOT
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .engine import set_reproducible
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .fpga_video_resize import area_3to2, bilinear_2to3, verify as verify_resize
from .publish_moderate_contours import encode
from .render_wavelet_game import close_pipe, video_check
from .publish_region_classes import motion_map


@torch.inference_mode()
def main(step,variant=None):
    configure_workspace();set_reproducible(2026100501);cv2.setNumThreads(2);contract_guard()
    name=variant or str(step)
    package=OUT/f'package-{name}';meta=json.loads((package/'verification.json').read_text())
    weights=package/'coefficients-padded.bin';assert file_hash(weights)==meta['sha256']
    net=IntegerB1(runtime_from_image(weights.read_bytes()))
    out=OUT/f'videos-{name}';out.mkdir(exist_ok=False)
    resize=verify_resize();write_json(out/'resize_verification.json',resize)
    clips=json.loads((ROOT/'clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    results=[]
    for clip in clips:
        if clip['id'] not in ('stairs','new_hallway','plaza'):continue
        assert file_hash(clip['clip'])==clip['clip_sha256']
        cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert 300<=count<=600 and cap.get(cv2.CAP_PROP_FPS)==30
        path=out/(clip['id']+'_integer_1080p.mp4');pipe=encode(path)
        started=time.perf_counter();n=0;previous=None;duplicates=0;motion=[];saved=[]
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                raw=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert raw.shape==(1080,1920,3)
                source=area_3to2(raw)
                q=net(torch.tensor(source.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))
                rgb=(q[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
                output=bilinear_2to3(rgb);pipe.stdin.write(output.tobytes())
                if previous is not None:
                    duplicates+=int(np.array_equal(previous[1],rgb))
                    if n%5==0:
                        mx,my,valid=motion_map(previous[0],source);row=dict(frame=n)
                        for key,old,current in [('source',previous[0],source),('integer',previous[1],rgb)]:
                            warped=cv2.remap(old.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                            values=np.abs(current.astype(np.float32)-warped).mean(2)[valid]
                            row[key]=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)))
                        motion.append(row)
                previous=(source,rgb)
                if n in (0,90,120,210,330,count-1):
                    Image.fromarray(output).save(out/f'{clip["id"]}_{n:03d}_integer1080.png')
                    Image.fromarray(raw).save(out/f'{clip["id"]}_{n:03d}_source1080.png')
                    saved.append(dict(frame=n,rgb720_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest()))
                n+=1
                if n%30==0:
                    elapsed=time.perf_counter()-started
                    write_json(out/'progress.json',dict(scene=clip['id'],frames=n,total=count,seconds=elapsed))
                    print('INTEGER_VIDEO',clip['id'],n,count,round(elapsed,2),flush=True)
        finally:
            cap.release();close_pipe(pipe)
        assert n==count and file_hash(weights)==meta['sha256']
        check=video_check(path,count)
        # Quantify encoding loss using real pre-encoding PNGs.
        encoded=cv2.VideoCapture(str(path));errors=[]
        for row in saved:
            encoded.set(cv2.CAP_PROP_POS_FRAMES,row['frame']);ok,bgr=encoded.read();assert ok
            reference=np.array(Image.open(out/f'{clip["id"]}_{row["frame"]:03d}_integer1080.png'))
            delta=np.abs(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.int16)-reference.astype(np.int16))
            errors.append(dict(frame=row['frame'],mae=float(delta.mean()),p99=float(np.percentile(delta,99))))
        encoded.release()
        results.append(dict(id=clip['id'],video=check,original=clip['clip'],original_sha256=clip['clip_sha256'],
                            seconds=time.perf_counter()-started,duplicates=duplicates,saved=saved,encoding=errors,motion_sample_every_5_adjacent_frames=motion))
        write_json(out/'results.json',dict(results=results,weight_sha256=meta['sha256'],chain='RGB1080 -> area720 -> bytes-reloaded INT8/Q17 B1 -> bilinear1080',
                                          postprocess=False,teacher_at_inference=False,actual_board=False,bit_exact_small_oracle=True))
    assert len(results)==3
    sources={r['id']:'../../'+Path(r['original']).relative_to(ROOT).as_posix() for r in results}
    page='''<!doctype html><meta charset="utf-8"><title>FPGA B1 实际整数包预览</title>
<style>body{background:#101923;color:#e9edf4;font:17px system-ui;margin:24px}video{width:100%;max-width:1400px;background:black}button,select{font:inherit;padding:8px;margin:5px}a{color:#9fd6ff}img{max-width:100%}.note{max-width:1100px;line-height:1.65}</style>
<h1>现有 FPGA B1：实际整数权重包输出</h1><p class="note">从导出的20 KiB权重传输文件重新读取，执行24层卷积、INT8/Q17、残差和整数缩放。没有额外调色、滤镜、人物分割或教师参与输出。这里是软件整数模拟，未实际烧录板子，也不代表时序和画质验收通过。</p>
<select id="scene" aria-label="场景"><option value="plaza">广场人物 · 15秒</option><option value="stairs">台阶 · 12秒</option><option value="new_hallway">走廊 · 12秒</option></select>
<select id="version" aria-label="版本"><option value="integer">FPGA整数小网模拟</option><option value="source">原片</option></select><button id="play">播放/暂停</button><button id="seek">定位3秒</button>
<p id="state"></p><video id="video" controls playsinline preload="metadata"></video>
<p><a href="../package-STEP/coefficients-padded.bin">20 KiB实际权重镜像</a> · <a href="../package-STEP/verification.json">参数包和整数验证</a> · <a href="results.json">视频验证</a></p>
<h2>同帧静态：原片 / 冻结目标 / 实际整数输出</h2><a href="../package-STEP/comparison.jpg"><img src="../package-STEP/comparison.jpg"></a>
<script>const sources=SOURCES;const v=document.getElementById('video'),scene=document.getElementById('scene'),version=document.getElementById('version');
function load(reset=false){const t=reset?0:v.currentTime,p=!v.paused;v.pause();v.src=version.value==='source'?sources[scene.value]:scene.value+'_integer_1080p.mp4';v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration-.05);if(p)v.play();document.getElementById('state').textContent=scene.selectedOptions[0].text+' / '+version.selectedOptions[0].text+' / '+v.videoWidth+'×'+v.videoHeight;};v.load();}
scene.onchange=()=>load(true);version.onchange=()=>load();document.getElementById('play').onclick=()=>v.paused?v.play():v.pause();document.getElementById('seek').onclick=()=>{v.currentTime=3;v.pause()};v.onerror=()=>document.getElementById('state').textContent='视频加载失败，请查看同目录MP4';load(true);</script>'''
    (out/'preview.html').write_text(page.replace('STEP',name).replace('SOURCES',json.dumps(sources)),encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    print('VIDEOS_COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=2000);parser.add_argument('--variant');a=parser.parse_args();main(a.step,a.variant)
