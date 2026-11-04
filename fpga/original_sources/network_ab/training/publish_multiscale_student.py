"""Export verification and full native-frame videos for a selected student."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn

from .multiscale_student import MultiscaleStudent, ConvBN, profile
from .train_multiscale_student import RUN
from .train_b1_distillation import as_tensor, to_rgb, read_image
from .fpga_video_resize import area_3to2, bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe, video_check
from .runtime import configure_workspace, file_hash, write_json, contract_guard


def load(run, size):
    folder = run/('train_'+size)
    report = json.loads((folder/'export.json').read_text())
    path = folder/'weights_fp32.pt'
    assert report['sha256'] == file_hash(path)
    model = MultiscaleStudent(size).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
    return model


def verify(root, run, size):
    out = run/('train_'+size)
    binding = json.loads((run/'binding.json').read_text())
    for p, digest in binding['code'].items():
        assert file_hash(p) == digest, p
    metrics = [json.loads(s) for s in (out/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in metrics] == list(range(1, 2001))
    assert all(np.isfinite(r['loss']) and r['grad_norm'] > 0 for r in metrics)
    counts = Counter(s['source'] for row in metrics for s in row['samples'])
    assert len(counts) == 7
    model = load(run, size)
    fused = copy.deepcopy(model)
    for layer in list(fused.modules()):
        if isinstance(layer, ConvBN):
            layer[1] = torch.nn.utils.fuse_conv_bn_eval(layer[1], layer[2]); layer[2] = nn.Identity()
    manifest = json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    errors = []; sample_errors = []; crop_errors = []
    with torch.inference_mode():
        for row in manifest['validation']:
            x = as_tensor(read_image(row['source']))[None].cuda()
            p = model(x)
            errors.append(float((p-fused(x)).abs().max()))
            saved = read_image(str(out/'validation_002000'/(row['id']+'.png')))
            sample_errors.append(int(np.abs(to_rgb(p[0]).astype(int)-saved.astype(int)).max()))
            # Aligned crop with sufficient halo: detects crop/resize/padding discrepancies.
            a, b, m, c = 384, 160, 96, 512
            crop = model(x[:, :, b:b+c, a:a+c])
            crop_errors.append(float((crop[:,:,m:-m,m:-m]-p[:,:,b+m:b+c-m,a+m:a+c-m]).abs().max()))
        info = profile(model, x)
    assert max(errors) < 5e-5 and max(sample_errors) == 0
    assert max(crop_errors) < 5e-5, crop_errors
    report = dict(status='FP32_CHECKS_PASS_NOT_STYLE_OR_DEPLOYMENT_ACCEPTANCE', size=size, steps=2000,
        source_counts=dict(counts), native_updates=sum(r['native'] for r in metrics),
        optimizer_seconds=sum(r['seconds'] for r in metrics), peak_mib=max(r['peak_mib'] for r in metrics),
        bn_fold_max_error=max(errors), image_reload_max_error=max(sample_errors), aligned_crop_max_error=max(crop_errors),
        profile=info, weights_sha256=file_hash(out/'weights_fp32.pt'), hardware_modified=False,
        training_launches=[json.loads(p.read_text()) for p in sorted(out.glob('launch_*.json'))])
    write_json(run/'verification.json', report)
    print('VERIFY_PASS', size, max(errors), max(crop_errors), flush=True)


def render(root, run, size):
    model = load(run, size)
    out = run/'videos'; out.mkdir(exist_ok=False)
    clips = json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    records = []
    for clip in clips:
        assert file_hash(clip['clip']) == clip['clip_sha256']
        assert 10 <= clip['validation']['seconds'] <= 20
        path = out/(clip['id']+'_multiscale_1080p.mp4')
        cap = cv2.VideoCapture(clip['clip']); pipe = encode_pipe(path, 1920, 1080, 30)
        n = 0; begin = time.perf_counter(); previous = None; duplicates = 0
        while True:
            ok, bgr = cap.read()
            if not ok: break
            assert bgr.shape == (1080, 1920, 3)
            rgb = area_3to2(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with torch.inference_mode(): result = to_rgb(model(as_tensor(rgb)[None].cuda())[0])
            full = bilinear_2to3(result); data = full.tobytes()
            duplicates += int(previous == data); previous = data
            pipe.stdin.write(data)
            if n in (0, 90, 120, 210, 359, 449):
                Image.fromarray(result).save(out/f'{clip["id"]}_{n:03d}_native720.png')
                Image.fromarray(full).save(out/f'{clip["id"]}_{n:03d}_preencode1080.png')
            n += 1
            if n % 90 == 0: print('MULTISCALE_VIDEO', clip['id'], n, round(time.perf_counter()-begin, 1), flush=True)
        cap.release(); close_pipe(pipe)
        assert n == clip['validation']['frames']
        records.append(dict(id=clip['id'], path=str(path), source=clip['clip'], source_sha256=clip['clip_sha256'],
            validation=video_check(path, n), model_calls=n, identical_adjacent_outputs=duplicates, seconds=time.perf_counter()-begin))
        write_json(out/'report.json', dict(status='COMPLETE' if len(records)==len(clips) else 'RENDERING', records=records,
            pipeline='1920x1080 -> integer area1280x720 -> MultiscaleStudent -> integer bilinear1920x1080',
            postprocess=False, source_blend=False, learned_rgb_residual_inside_model=True, temporal_blend=False, frame_repeat=False, crf=16,
            weights_sha256=file_hash(run/('train_'+size)/'weights_fp32.pt')))
    print('VIDEOS_COMPLETE', flush=True)


def page(root, run, size):
    manifest = json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    image = Image.new('RGB', (960, 946), '#171c24'); draw = ImageDraw.Draw(image)
    for i, label in enumerate(('教师', '之前C8 / 2000步', '本轮多尺度 / 2000步')):
        draw.text((320*i+8, 8), label, fill='white', font=font)
    for i, (key, xy, label) in enumerate([('val_stairs_090',(424,320),'石柱'),('val_stairs_090',(180,430),'植物'),('val_plaza_090',(80,448),'人物')]):
        row = next(r for r in manifest['validation'] if r['id']==key)
        paths = [Path(row['target']), root/'runs/fullres-c8-stylepush-color-20261003/train/validation_002000'/f'{key}.png',
                 run/('train_'+size)/'validation_002000'/f'{key}.png']
        y = 42+i*288; draw.text((8,y), label, font=font, fill='#bfd0e9')
        x0,y0 = xy
        for j,p in enumerate(paths): image.paste(Image.open(p).crop((x0,y0,x0+320,y0+256)),(j*320,y+28))
    draw.text((8,918),'同帧同位置、720p原生像素裁切；未缩放或锐化。',font=font,fill='#bfd0e9')
    image.save(run/'teacher_c8_multiscale.png')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多尺度学生：实际连续视频</title><style>body{max-width:1200px;margin:24px auto;padding:0 18px;background:#151b24;color:#edf3fa;font:17px/1.7 system-ui}a{color:#a9d1ff}video{width:100%;max-height:76vh;background:black}img{max-width:100%;height:auto}button,select{font:inherit;padding:8px;margin:5px;color:white;background:#2e425a;border:1px solid #789}.expanded{position:fixed;inset:0;width:100vw;height:100vh;max-height:none;z-index:1000;object-fit:contain}#close{display:none;position:fixed;top:10px;right:10px;z-index:1001}.large #close{display:block}#error{color:#ffb2ac}</style>
<h1>多尺度学生：实际训练输出</h1><p>三个训练区域短拟合600步后，继续7个来源、95帧对训练2000步。这里的场景未进入本轮训练，但之前看过，属于回归检查。没有额外锐化、脸部贴回或时间混帧。</p>
<p>走廊12秒、台阶12秒、广场15秒；1080→720网络处理→1080输出。新网络尚未完成FPGA部署。</p>
<label>场景<select id="scene"><option value="stairs">台阶 / 12秒</option><option value="new_hallway">走廊 / 12秒</option><option value="plaza">广场 / 15秒</option></select></label>
<label>版本<select id="version"><option value="new">本轮多尺度</option><option value="teacher">教师</option><option value="old">之前C8</option><option value="source">原片</option></select></label><button id="play">播放 / 暂停</button><button id="seek">定位3秒</button><button id="expand">放大观看</button>
<div id="label"></div><video controls playsinline loop preload="metadata"></video><button id="close">返回对照</button><p id="error"></p>
<p>教师 / 之前C8 / 本轮，点击看原尺寸。</p><a href="teacher_c8_multiscale.png"><img src="teacher_c8_multiscale.png" alt="教师、旧学生、新学生的石柱植物人物原尺寸比较"></a>
<p><a href="RESULTS.md">结果与限制</a> · <a href="train_SIZE/weights_fp32.pt">本轮真实FP32权重</a> · <a href="video_diagnostics.json">完整连续帧测量</a></p>
<script>const v=document.querySelector('video'),scene=document.querySelector('#scene'),version=document.querySelector('#version');function load(keep){let t=keep?v.currentTime:0,playing=!v.paused,id=scene.value,k=version.value;v.pause();const path=k==='new'?'videos/'+id+'_multiscale_1080p.mp4':k==='old'?'../fullres-c8-stylepush-color-20261003/videos/'+id+'_c8_1080p.mp4':k==='teacher'?'../b1-distillation-20261003/videos-2000/'+id+'_teacher_1080p.mp4':'../clear-style-control-20261003/video_sources/'+id+'_source_1080p.mp4';document.querySelector('#error').textContent='';document.querySelector('#label').textContent=scene.options[scene.selectedIndex].text+' / '+version.options[version.selectedIndex].text;v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration-.1);if(playing)v.play().catch(()=>{})};v.src=path}scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();document.querySelector('#seek').onclick=()=>{v.pause();v.currentTime=3};function large(on){v.classList.toggle('expanded',on);document.body.classList.toggle('large',on)}document.querySelector('#expand').onclick=()=>large(true);document.querySelector('#close').onclick=()=>large(false);document.addEventListener('keydown',e=>{if(e.key==='Escape')large(false)});v.onerror=()=>document.querySelector('#error').textContent='视频读取失败';load(false);</script></html>'''
    (run/'index.html').write_text(html.replace('train_SIZE','train_'+size), encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['verify','render','diagnostics','page']); ap.add_argument('--size',choices=['small','medium'],required=True)
    args=ap.parse_args(); root=configure_workspace(); run=root/'runs'/RUN
    contract_guard(); torch.set_num_threads(4); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.mode=='diagnostics':
        # Reuse exactly the prior continuous-frame diagnostic implementation;
        # configure its reference run in this process without editing that file.
        from . import publish_fullres_c8_stylepush as diagnostic
        diagnostic.PREVIOUS='fullres-c8-stylepush-color-20261003'
        diagnostic.diagnostics(root,run)
    else: globals()[args.mode](root,run,args.size)


if __name__=='__main__':main()
