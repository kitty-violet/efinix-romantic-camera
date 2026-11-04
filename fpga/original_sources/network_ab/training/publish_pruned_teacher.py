"""Verify and publish real recovered-pruning outputs, without surface guard."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .prune_animation_teacher import RUN, build_pruned, profile
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .fpga_video_resize import area_3to2,bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe,video_check
from .runtime import configure_workspace,file_hash,write_json,contract_guard


def load(root,run,candidate):
    folder=run/(candidate+'_recovery');p=folder/'pruned_teacher_fp32.pt'
    report=json.loads((folder/'export.json').read_text());assert file_hash(p)==report['sha256']
    state=torch.load(p,map_location='cpu',weights_only=True)
    net=build_pruned(root,state['indices']).cuda().eval();net.load_state_dict(state['generator'])
    return net


def verify(root,run,candidate):
    folder=run/(candidate+'_recovery');binding=json.loads((folder/'binding.json').read_text())
    for p,h in binding['code'].items():assert file_hash(p)==h,p
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in rows]==list(range(1,301))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 and r['native_frames']==2 for r in rows)
    counts=Counter(r['source'] for r in rows);assert len(counts)==7
    assert json.loads((folder/'launch_000150.json').read_text())['step']==150
    model=load(root,run,candidate);manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));errors=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            x=as_tensor(read_image(row['source']))[None].cuda();output=to_rgb(model(x)[0])
            saved=read_image(str(folder/'validation_000300'/(row['id']+'.png')))
            errors.append(int(np.abs(output.astype(int)-saved.astype(int)).max()))
    assert max(errors)==0
    write_json(folder/'verification.json',dict(status='FP32_RELOAD_AND_TRAINING_CHECKED_NOT_HARDWARE_ACCEPTANCE',
        steps=300,real_native720_frames=600,source_counts=dict(counts),optimizer_seconds=sum(r['seconds'] for r in rows),
        peak_mib=max(r['peak_mib'] for r in rows),reload_image_error=max(errors),profile=profile(model,x),
        weights_sha256=file_hash(folder/'pruned_teacher_fp32.pt'),inference_guard=False,quantized=False,hardware_modified=False))
    print('VERIFY_PASS',candidate,flush=True)


def render(root,run,candidate):
    model=load(root,run,candidate);out=run/('videos_'+candidate);out.mkdir(exist_ok=False)
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    records=[]
    for clip in clips:
        assert file_hash(clip['clip'])==clip['clip_sha256'] and 10<=clip['validation']['seconds']<=20
        p=out/(clip['id']+'_pruned_1080p.mp4');cap=cv2.VideoCapture(clip['clip']);pipe=encode_pipe(p,1920,1080,30)
        count=0;previous=None;duplicates=0;begin=time.perf_counter()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert bgr.shape==(1080,1920,3)
            rgb=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            with torch.inference_mode():result=to_rgb(model(as_tensor(rgb)[None].cuda())[0])
            full=bilinear_2to3(result);data=full.tobytes();duplicates+=int(previous==data);previous=data;pipe.stdin.write(data)
            if count in (0,90,120,210,359,449):
                Image.fromarray(result).save(out/f'{clip["id"]}_{count:03d}_native720.png')
                Image.fromarray(full).save(out/f'{clip["id"]}_{count:03d}_preencode1080.png')
            count+=1
            if count%90==0:print('PRUNED_VIDEO',candidate,clip['id'],count,round(time.perf_counter()-begin,1),flush=True)
        cap.release();close_pipe(pipe);assert count==clip['validation']['frames']
        records.append(dict(id=clip['id'],path=str(p),source=clip['clip'],source_sha256=clip['clip_sha256'],validation=video_check(p,count),
                            model_calls=count,identical_adjacent_outputs=duplicates,seconds=time.perf_counter()-begin))
        write_json(out/'report.json',dict(status='COMPLETE' if len(records)==len(clips) else 'RENDERING',records=records,
            pipeline='1080 integer area -> 720 recovered-pruned neural network -> integer bilinear1080',
            surface_guard=False,source_blend=False,temporal_blend=False,frame_repeat=False,crf=16,
            weights_sha256=file_hash(run/(candidate+'_recovery')/'pruned_teacher_fp32.pt')))
    print('VIDEOS_COMPLETE',candidate,flush=True)


def diagnostics(root,run,candidate):
    from . import publish_fullres_c8_stylepush as old
    # This existing function reads run/videos/report.json and writes a diagnostic
    # beside it. Use an isolated evaluation directory with references, no videos copied.
    evaluation=run/('evaluation_'+candidate);(evaluation/'videos').mkdir(parents=True,exist_ok=False)
    report=json.loads((run/('videos_'+candidate)/'report.json').read_text())
    write_json(evaluation/'videos/report.json',report)
    old.PREVIOUS='fullres-c8-stylepush-color-20261003';old.diagnostics(root,evaluation)


def page(root,run,candidate):
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    panel=Image.new('RGB',(960,946),'#171c24');draw=ImageDraw.Draw(panel)
    labels=('原教师目标','之前C8 / 2000步','教师剪枝后 / 恢复300步')
    for j,label in enumerate(labels):draw.text((320*j+8,8),label,font=font,fill='white')
    for i,(key,xy,label) in enumerate([('val_stairs_090',(424,320),'石柱'),('val_stairs_090',(180,430),'植物'),('val_plaza_090',(80,448),'人物')]):
        row=next(r for r in manifest['validation'] if r['id']==key)
        paths=[Path(row['target']),root/'runs/fullres-c8-stylepush-color-20261003/train/validation_002000'/f'{key}.png',
               run/(candidate+'_recovery')/'validation_000300'/f'{key}.png']
        y=42+i*288;draw.text((8,y),label,font=font,fill='#bcd2eb');x0,y0=xy
        for j,p in enumerate(paths):panel.paste(Image.open(p).crop((x0,y0,x0+320,y0+256)),(j*320,y+28))
    draw.text((8,918),'同帧同位置、原生像素直接裁切；剪枝版不加锐化、表面修正或时间混帧。',font=font,fill='#bcd2eb')
    panel.save(run/(candidate+'_comparison.png'))
    if not (run/('videos_'+candidate)/'report.json').exists():
        still='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>更小剪枝档位：静态对照</title>
<style>body{max-width:1100px;margin:28px auto;padding:0 20px;background:#151b24;color:#edf3fa;font:18px/1.8 system-ui}a{color:#abd4ff}img{max-width:100%}</style>
<h1>更小剪枝档位：静态对照</h1><p>CANDIDATE已实际恢复300步并导出权重。本页只有静态对照，没有生成这一档的视频；判断与限制见实验报告。</p>
<p><a href="index_keep50.html">返回半通道版完整视频</a> · <a href="RESULTS.md">结果与限制</a> · <a href="CANDIDATE_recovery/pruned_teacher_fp32.pt">真实FP32权重</a></p>
<a href="CANDIDATE_comparison.png"><img src="CANDIDATE_comparison.png" alt="教师、旧C8、更小剪枝版同帧对照"></a></html>'''
        (run/('index_'+candidate+'.html')).write_text(still.replace('CANDIDATE',candidate),encoding='utf-8')
        return
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>教师通道剪枝：实际恢复效果</title>
<style>body{max-width:1200px;margin:24px auto;padding:0 18px;background:#151b24;color:#edf3fa;font:17px/1.7 system-ui}a{color:#a9d1ff}video{width:100%;max-height:76vh;background:black}img{max-width:100%;height:auto}select,button{font:inherit;padding:8px;margin:5px;color:white;background:#2e425a;border:1px solid #789}.expanded{position:fixed;inset:0;width:100vw;height:100vh;max-height:none;z-index:1000;object-fit:contain}#close{display:none;position:fixed;top:10px;right:10px;z-index:1001}.large #close{display:block}#error{color:#ffb2ac}</style>
<h1>从教师实际删除通道，再恢复训练</h1><p>CANDIDATE：恢复300步，每步两张完整720p帧，7个训练来源。这里是已看过但来源隔离的回归视频，不是盲测。剪枝版直接来自真实权重，未叠加表面修正、锐化或时间混帧。</p>
<p>走廊12秒、台阶12秒、广场15秒；1080→720处理→1080输出。剪枝图仍含动态归一化和注意力，尚不能直接作为B1权重上板。</p>
<label>场景<select id="scene"><option value="stairs">台阶 / 12秒</option><option value="new_hallway">走廊 / 12秒</option><option value="plaza">广场 / 15秒</option></select></label><label>版本<select id="version"><option value="new">教师剪枝恢复版</option><option value="teacher">原教师目标</option><option value="old">之前C8</option><option value="source">原片</option></select></label><button id="play">播放 / 暂停</button><button id="seek">定位3秒</button><button id="expand">放大观看</button><div id="label"></div><video controls playsinline loop preload="metadata"></video><button id="close">返回对照</button><p id="error"></p>
<p>原教师 / 之前C8 / 教师剪枝恢复版，点击看原尺寸。</p><a href="CANDIDATE_comparison.png"><img src="CANDIDATE_comparison.png" alt="教师、旧学生和剪枝恢复版同帧比较"></a>
<p><a href="RESULTS.md">结果与限制</a> · <a href="CANDIDATE_recovery/pruned_teacher_fp32.pt">真实FP32权重</a> · <a href="evaluation_CANDIDATE/video_diagnostics.json">完整连续帧测量</a></p>
<p><a href="index_keep25.html">进一步压缩到四分之一通道的静态对照</a></p>
<p>广场11.97秒的近处人物：原片／教师／半通道剪枝版，同帧视频解码比较。</p><a href="near_faces_frame359.png"><img src="near_faces_frame359.png" alt="近处人物的原片、教师、剪枝版对照"></a>
<script>const v=document.querySelector('video'),scene=document.querySelector('#scene'),version=document.querySelector('#version');function load(keep){let t=keep?v.currentTime:0,playing=!v.paused,id=scene.value,k=version.value;v.pause();const p=k==='new'?'videos_CANDIDATE/'+id+'_pruned_1080p.mp4':k==='old'?'../fullres-c8-stylepush-color-20261003/videos/'+id+'_c8_1080p.mp4':k==='teacher'?'../b1-distillation-20261003/videos-2000/'+id+'_teacher_1080p.mp4':'../clear-style-control-20261003/video_sources/'+id+'_source_1080p.mp4';document.querySelector('#error').textContent='';document.querySelector('#label').textContent=scene.options[scene.selectedIndex].text+' / '+version.options[version.selectedIndex].text;v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration-.1);if(playing)v.play().catch(()=>{})};v.src=p}scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();document.querySelector('#seek').onclick=()=>{v.pause();v.currentTime=3};function large(on){v.classList.toggle('expanded',on);document.body.classList.toggle('large',on)}document.querySelector('#expand').onclick=()=>large(true);document.querySelector('#close').onclick=()=>large(false);document.addEventListener('keydown',e=>{if(e.key==='Escape')large(false)});v.onerror=()=>document.querySelector('#error').textContent='视频读取失败';load(false);</script></html>'''
    (run/('index_'+candidate+'.html')).write_text(html.replace('CANDIDATE',candidate),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','render','diagnostics','page']);ap.add_argument('--candidate',choices=['keep25','keep50'],required=True);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[args.mode](root,run,args.candidate)


if __name__=='__main__':main()
