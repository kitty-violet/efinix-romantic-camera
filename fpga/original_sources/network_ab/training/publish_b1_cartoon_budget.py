"""Actual B1-only weights, before/after clips and traceable playback page."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from network_ab.contract_torch import build_fp32_model
from .train_b1_cartoon_budget import RUN
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .fpga_video_resize import area_3to2,bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe,video_check


def load(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=build_fp32_model().cuda().eval();net.load_state_dict(state['model']);return net


def export(root,run,step):
    path=run/f'train/checkpoint_{step:06d}.pt';state=torch.load(path,map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    assert state['step']==step
    rows=[json.loads(s) for s in (run/'train/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,step+1))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 for r in rows)
    _,plan=contract_guard();out=run/f'final-{step}';out.mkdir(exist_ok=False)
    payload=dict(schema='exact-b1-cartoon-budget-fp32-v1',model={k:v.cpu() for k,v in state['model'].items()},step=step,
        contract_canonical_sha256=plan['contract_sha256'],contract_file_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
        model_id=plan['model_id'],training_binding=state['binding'],training_config=state['config'],
        checkpoint_sha256=file_hash(path),dtype='FP32',quantized=False,postprocess=False,temporal_state=False,
        training_only_judge_excluded=True,hardware_release=False)
    weight=out/'b1_cartoon_fp32.pt';torch.save(payload,weight)
    net=load(weight);reference=build_fp32_model().cuda().eval();reference.load_state_dict(state['model'])
    folded=copy.deepcopy(net)
    for seq in folded.operators.values():
        for i,layer in enumerate(seq):
            if isinstance(layer,torch.nn.BatchNorm2d):seq[i-1]=torch.nn.utils.fuse_conv_bn_eval(seq[i-1],layer);seq[i]=torch.nn.Identity()
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));checks=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            x=as_tensor(read_image(row['source']))[None].cuda();p=net(x)
            err=float((reference(x)-p).abs().max());ferr=float((folded(x)-p).abs().max())
            saved=read_image(str(run/f'train/validation_{step:06d}'/(row['id']+'.png')))
            pngerr=int(np.abs(saved.astype(int)-to_rgb(p[0]).astype(int)).max())
            assert err==0 and ferr<1e-4 and pngerr==0
            checks.append(dict(id=row['id'],reload_max=err,bn_fold_max=ferr,png_max=pngerr))
    sources=Counter(s['source'] for r in rows for s in r['samples'])
    write_json(out/'verification.json',dict(status='FP32_EXPORT_PASS_NOT_INT8_DEPLOYMENT',step=step,checks=checks,
        weights_sha256=file_hash(weight),bytes=weight.stat().st_size,parameters=sum(p.numel() for p in net.parameters()),
        macs_per_frame=plan['macs_per_frame'],sources=dict(sources),native_updates=sum(r['native'] for r in rows),
        optimizer_seconds=sum(r['seconds'] for r in rows),peak_mib=max(r['peak_mib'] for r in rows),
        independent_resume_at=[int(p.stem.split('_')[-1]) for p in (run/'train').glob('launch_*.json') if p.stem!='launch_000000']))
    (out/'INFERENCE.md').write_text(f'''# 当前B1原图推理

使用 `network_ab.contract_torch.build_fp32_model()`，加载 `b1_cartoon_fp32.pt` 的 `model` 字段并 `.eval()`。
输入 RGB uint8 经 `(x-128)/128` 为 NCHW float32；网络内部原生1280×720；输出乘128加128、round、clamp到uint8。
1080输入调用 `fpga_video_resize.area_3to2`，显示调用 `bilinear_2to3`。无教师、判别器、光流、时间缓存、输出滤镜或源图融合。

本轮额外训练{step}步。卷积/BN层名与当前未修改合同一致；部署时BN可折叠。FP32不是INT8系数，仍需量化、Q17运行参数、整数/RTL对拍。
''',encoding='utf-8')
    print('B1_EXPORT',step,file_hash(weight),flush=True)


def pictures(root,run,step):
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    old=root/'runs/b1-cartoon-budget-20261004/train/validation_000000'
    new=run/f'train/validation_{step:06d}'
    canvas=Image.new('RGB',(1280,962),'#151c25');d=ImageDraw.Draw(canvas)
    labels=['原片','教师目标','旧 B1（2000步）',f'同一 B1 新训 {step}步']
    for i,l in enumerate(labels):d.text((320*i+5,7),l,font=font,fill='white')
    for j,(key,xy,label) in enumerate([('val_stairs_090',(424,320),'石柱与台阶'),('val_stairs_090',(150,430),'草与台阶'),('val_plaza_090',(80,400),'人物与地面')]):
        row=next(r for r in manifest['validation'] if r['id']==key);x,y=xy;top=38+j*297
        d.text((8,top),label,font=font,fill='#a9cce9')
        for i,p in enumerate([Path(row['source']),Path(row['target']),old/(key+'.png'),new/(key+'.png')]):
            canvas.paste(Image.open(p).crop((x,y,x+320,y+264)),(i*320,top+26))
    d.text((6,932),'原生720像素裁切；没有锐化、模糊、贴脸或混原片。',font=font,fill='#a9cce9');canvas.save(run/'comparison.png')
    # Full native pictures also linked; thumbnails cannot establish sharpness.
    for row in manifest['validation']:
        key=row['id'];board=Image.new('RGB',(1280*3,752),'#151c25');d=ImageDraw.Draw(board)
        for i,(p,label) in enumerate([(row['source'],'原片'),(row['target'],'教师'),(new/(key+'.png'),'本轮原生720输出')]):
            d.text((1280*i+8,4),label,font=font,fill='white');board.paste(Image.open(p),(1280*i,32))
        board.save(run/(key+'_full.png'))


def render(root,run,step):
    net=load(run/f'final-{step}/b1_cartoon_fp32.pt')
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));train={p['source'] for p in manifest['pairs']}
    folder=run/'videos';folder.mkdir(exist_ok=False);records=[]
    # Adapt the historic report format for the shared all-frame diagnostic, without changing it.
    old=json.loads((root/'runs/b1-distillation-20261003/videos-2000/report.json').read_text())
    previous=run/'previous/videos';previous.mkdir(parents=True)
    write_json(previous/'report.json',dict(status='COMPLETE',records=[dict(id=r['id'],path=r['student'],source=r['source'],source_sha256=r['clip_sha256'],validation=r['student_validation']) for r in old['records']]))
    for clip in clips:
        assert file_hash(clip['clip'])==clip['clip_sha256'] and clip['source'] not in train
        cap=cv2.VideoCapture(clip['clip']);path=folder/(clip['id']+'_1080p.mp4');pipe=encode_pipe(path,1920,1080,30)
        n=0;last=None;dups=0;start=time.perf_counter()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            with torch.inference_mode():native=to_rgb(net(as_tensor(source)[None].cuda())[0])
            full=bilinear_2to3(native);raw=full.tobytes();dups+=int(raw==last);last=raw;pipe.stdin.write(raw)
            if n in (90,120,210,359):
                Image.fromarray(native).save(folder/f'{clip["id"]}_{n:03d}_native720.png')
                Image.fromarray(full).save(folder/f'{clip["id"]}_{n:03d}_preencode1080.png')
            n+=1
            if n%90==0:print('B1_VIDEO',clip['id'],n,round(time.perf_counter()-start,2),flush=True)
        cap.release();close_pipe(pipe);assert n==clip['validation']['frames']
        check=video_check(path,n)
        records.append(dict(id=clip['id'],path=str(path),source=clip['clip'],source_sha256=clip['clip_sha256'],validation=check,model_calls=n,identical_adjacent_outputs=dups))
        write_json(folder/'report.json',dict(status='COMPLETE' if len(records)==3 else 'RUNNING',records=records,
            weights_sha256=file_hash(run/f'final-{step}/b1_cartoon_fp32.pt'),step=step,pipeline='1080 integer area ->720 unchanged B1 ->1080 integer bilinear',
            postprocessing=False,frame_blending=False,source_blending=False,crf=16))
    print('B1_VIDEOS_COMPLETE',flush=True)


def diagnostics(root,run,step):
    from . import publish_fullres_c8_stylepush as shared
    shared.PREVIOUS=RUN+'/previous';shared.diagnostics(root,run)


def page(root,run,step):
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>当前FPGA网络：动画训练实测</title>
<style>body{max-width:1440px;margin:24px auto;padding:0 18px;background:#151b24;color:#edf3fa;font:17px/1.7 system-ui}a{color:#aed7ff}select,button{font:inherit;padding:7px;margin:5px;background:#273648;color:white;border:1px solid #748ba4;border-radius:5px}video{width:100%;max-height:78vh;background:#000}img{max-width:100%}.controls{position:sticky;top:0;background:#151b24;padding:8px;z-index:2}.notice{background:#263442;padding:12px}</style>
<h1>直接训练当前 FPGA 的 B1 网络</h1><p>相同24个卷积、8/16/32通道、0.967G MAC。保留现有网络图，用不同的训练约束学习色块、轮廓和连续帧稳定；额外训练STEP步。判别器只在训练中使用。</p>
<p class="notice">这里展示实际权重的直接输出。是否达到动画观感请同时看细节和连续视频；FP32实验完成不等于INT8或上板验收。<a href="RESULTS.md">实测结论与限制</a></p>
<p>走廊12秒、台阶12秒、广场人物15秒，30fps真实逐帧推理。1080→720→1080。教师和旧B1使用此前同一套视频，13.9G参照用于观察压缩损失。</p>
<div class="controls"><label>场景<select id="scene"><option value="stairs">台阶 / 12秒</option><option value="new_hallway">走廊 / 12秒</option><option value="plaza">广场人物 / 15秒</option></select></label><label>版本<select id="version"><option value="new">本轮 B1 / STEP步</option><option value="previous">旧 B1 / 2000步</option><option value="repair">13.9G 修正目标续训参照</option><option value="teacher">教师目标</option><option value="source">原片</option></select></label><button id="play">播放 / 暂停</button><button id="seek">定位3秒</button><span id="loading"></span></div><video controls playsinline preload="metadata"></video><p id="error"></p>
<p>以下图片四列：原片／教师／旧 B1／本轮。点击查看原尺寸。</p><a href="comparison.png"><img src="comparison.png" alt="当前B1原生像素对比"></a>
<p>完整720帧：<a href="val_stairs_090_full.png">台阶与植物</a> · <a href="val_new_hallway_210_full.png">走廊墙面</a> · <a href="val_plaza_210_full.png">人物广场</a></p>
<p><a href="final-STEP/b1_cartoon_fp32.pt">真实 B1 FP32 权重</a> · <a href="final-STEP/INFERENCE.md">加载方法</a> · <a href="hardware_boundary.json">网络与硬件边界</a> · <a href="video_diagnostics.json">全部连续帧测量</a></p>
<script>const v=document.querySelector('video'),scene=document.querySelector('#scene'),version=document.querySelector('#version'),seek=document.querySelector('#seek'),loading=document.querySelector('#loading');let pendingTime=0,busy=false,wantPlay=false,serial=0;
function load(keep){const n=++serial;if(!busy){pendingTime=keep?v.currentTime:0;wantPlay=!v.paused}else if(!keep)pendingTime=0;v.pause();busy=true;seek.disabled=true;loading.textContent='读取视频…';const id=scene.value,k=version.value;const url=k==='new'?'videos/'+id+'_1080p.mp4':k==='previous'?'../b1-distillation-20261003/videos-2000/'+id+'_b1_1080p.mp4':k==='repair'?'../branch-function-audit-20261004/reference/videos/'+id+'_1080p.mp4':k==='teacher'?'../b1-distillation-20261003/videos-2000/'+id+'_teacher_1080p.mp4':'../clear-style-control-20261003/video_sources/'+id+'_source_1080p.mp4';v.onloadedmetadata=()=>{if(n!==serial)return;v.currentTime=Math.min(pendingTime,v.duration-.05);busy=false;seek.disabled=false;loading.textContent='';if(wantPlay)v.play().catch(()=>{})};v.onerror=()=>{busy=false;loading.textContent='';document.querySelector('#error').textContent='视频读取失败，请使用本地文件链接。'};document.querySelector('#error').textContent='';v.src=url;v.load()}
scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>{if(!busy){if(v.paused){wantPlay=true;v.play().catch(()=>{})}else{wantPlay=false;v.pause()}}};seek.onclick=()=>{if(!busy){v.pause();wantPlay=false;v.currentTime=3}};load(false);</script></html>'''
    (run/'review.html').write_text(html.replace('STEP',str(step)),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['export','pictures','render','diagnostics','page']);ap.add_argument('--step',type=int,default=2000);a=ap.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100407);cv2.setNumThreads(2)
    globals()[a.action](root,root/'runs'/RUN,a.step)


if __name__=='__main__':main()
