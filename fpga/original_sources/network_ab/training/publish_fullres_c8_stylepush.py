"""Publish actual style-push weights and continuous videos, without output effects."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .train_fullres_c8_stylepush import RUN,PREVIOUS
from .publish_fullres_c8_multiscene import load,render
from .fpga_video_resize import area_3to2
from .train_b1_distillation import as_tensor,read_image
from .runtime import configure_workspace,contract_guard,file_hash,write_json


def verify(root,run):
    rows=[json.loads(s) for s in (run/'train/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,2001))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 for r in rows)
    counts=Counter(s['source'] for r in rows for s in r['samples']);assert len(counts)==7
    assert json.loads((run/'train/launch_000250.json').read_text())['initial_step']==250
    model=load(run);fused=copy.deepcopy(model)
    for seq in fused.body:seq[1]=torch.nn.utils.fuse_conv_bn_eval(seq[1],seq[2]);seq[2]=nn.Identity()
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));errors=[]
    with torch.inference_mode():
        for r in manifest['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda()
            errors.append(float((model(x)-fused(x)).abs().max()))
    assert max(errors)<5e-5
    write_json(run/'verification.json',dict(status='FP32_PASS_NOT_PRODUCTION',steps=2000,
        source_counts=dict(counts),native_steps=sum(r['native'] for r in rows),
        optimizer_seconds=sum(r['seconds'] for r in rows),peak_mib=max(r['peak_mib'] for r in rows),
        bn_fold_max_error=max(errors),weights_sha256=file_hash(run/'final/fullres_c8_fp32.pt'),
        resume_process_at=250,hardware_modified=False,qat=False))
    print('VERIFY_PASS',max(errors),flush=True)


def diagnostics(root,run):
    report=json.loads((run/'videos/report.json').read_text());assert report['status']=='COMPLETE'
    oldreport=json.loads((root/'runs'/PREVIOUS/'videos/report.json').read_text())
    teacherreport=json.loads((root/'runs/b1-distillation-20261003/videos-2000/report.json').read_text())
    # Faster equivalent area reduction only for measurement. Production rendering
    # keeps the previously verified integer implementation.
    reduce=lambda rgb:cv2.resize(rgb,(1280,720),interpolation=cv2.INTER_AREA)
    rng=np.random.default_rng(103);resize_checks=0
    for _ in range(3):
        a=rng.integers(0,256,(1080,1920,3),dtype=np.uint8)
        assert np.array_equal(reduce(a),area_3to2(a));resize_checks+=1
    allrows=[]
    yy,xx=np.mgrid[:720,:1280].astype(np.float32)
    for clip in report['records']:
        name=clip['id'];old=next(r for r in oldreport['records'] if r['id']==name)
        teacher=next(r for r in teacherreport['records'] if r['id']==name)
        paths=dict(source=clip['source'],new=clip['path'],previous=old['path'],
            teacher=str(root/f'runs/b1-distillation-20261003/videos-2000/{name}_teacher_1080p.mp4'))
        expected=dict(source=clip['source_sha256'],new=clip['validation']['sha256'],
            previous=old['validation']['sha256'],teacher=teacher['teacher_validation']['sha256'])
        assert old['source_sha256']==clip['source_sha256']==teacher['clip_sha256']
        for k,p in paths.items():assert file_hash(p)==expected[k]
        caps={k:cv2.VideoCapture(p) for k,p in paths.items()};previous=None;n=0;coverage=[]
        values={k:dict(temporal=[],change=[],new_white=0,material_dot=0.,material_energy=0.,teacher_energy=0.) for k in ('new','previous','teacher')}
        while True:
            raw={k:c.read() for k,c in caps.items()}
            if not raw['source'][0]:assert not any(v[0] for v in raw.values());break
            assert all(v[0] for v in raw.values())
            rawrgb={k:cv2.cvtColor(v[1],cv2.COLOR_BGR2RGB) for k,v in raw.items()}
            rgb={k:reduce(v) for k,v in rawrgb.items()}
            if n in (0,90):
                for k in rgb:assert np.array_equal(rgb[k],area_3to2(rawrgb[k]));resize_checks+=1
            gray={k:cv2.cvtColor(v,cv2.COLOR_RGB2GRAY).astype(np.float32) for k,v in rgb.items()}
            hp=lambda v:v-cv2.blur(v,(9,9),borderType=cv2.BORDER_REPLICATE)
            teacher_delta=hp(gray['teacher'])-hp(gray['source'])
            if previous is not None:
                cur=cv2.resize(gray['source'],(640,360));past=cv2.resize(previous['source'],(640,360))
                back=cv2.calcOpticalFlowFarneback(cur,past,None,.5,3,21,3,5,1.2,0)
                forward=cv2.calcOpticalFlowFarneback(past,cur,None,.5,3,21,3,5,1.2,0)
                flow=cv2.resize(back,(1280,720))*2;fw=cv2.resize(forward,(1280,720))*2
                mx,my=xx+flow[:,:,0],yy+flow[:,:,1]
                sample=lambda v:cv2.remap(v,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                valid=(np.abs(gray['source']-sample(previous['source']))<8)&(np.linalg.norm(flow+sample(fw),axis=2)<1.5)&(mx>=1)&(mx<1278)&(my>=1)&(my<718)
                assert valid.any();coverage.append(float(valid.mean()))
            for k,d in values.items():
                effect=gray[k]-gray['source'];band=hp(gray[k])-hp(gray['source'])
                if previous is not None:d['temporal'].append(float(np.abs(effect-sample(previous[k]-previous['source']))[valid].mean()))
                d['change'].append(float(np.abs(rgb[k].astype(np.float32)-rgb['source']).mean()))
                d['new_white']+=int(((rgb[k].min(2)>245)&(rgb['source'].max(2)<220)).sum())
                d['material_dot']+=float((band*teacher_delta).sum(dtype=np.float64))
                d['material_energy']+=float((band*band).sum(dtype=np.float64))
                d['teacher_energy']+=float((teacher_delta*teacher_delta).sum(dtype=np.float64))
            previous=gray;n+=1
            if n%120==0:print('STYLEPUSH_DIAGNOSTIC',name,n,flush=True)
        for c in caps.values():c.release()
        assert n==clip['validation']['frames'];summary={}
        for k,d in values.items():
            assert len(d['temporal'])==n-1
            summary[k]=dict(temporal_mean=float(np.mean(d['temporal'])),temporal_p95=float(np.percentile(d['temporal'],95)),
                source_change_rgb=float(np.mean(d['change'])),newly_white_fraction=d['new_white']/(n*1280*720),
                material_projection=d['material_dot']/max(1e-9,d['teacher_energy']),
                material_cosine=d['material_dot']/max(1e-9,(d['material_energy']*d['teacher_energy'])**.5))
        allrows.append(dict(id=name,frames=n,variants=summary,flow_valid_fraction=float(np.mean(coverage)),input_hashes=expected))
        write_json(run/'video_diagnostics.json',dict(status='COMPLETE' if len(allrows)==3 else 'MEASURING',records=allrows,
            bitexact_area_checks=resize_checks,grid=[1280,720],flow_estimation_grid=[640,360],
            limitations=['Style projection measures correlation with teacher texture change, not perceptual acceptance.',
                'Low temporal error can result from weak style. Thin plants and occlusions may be excluded by flow mask.',
                'New-white thresholds exclude original bright pixels; not a general exposure detector.']))
    print('STYLEPUSH_DIAGNOSTICS_COMPLETE',flush=True)


def page(root,run):
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>加强材质风格 · C8 实际训练结果</title>
<style>body{max-width:1280px;margin:24px auto;padding:0 18px;background:#171c23;color:#edf3f7;font:16px/1.65 system-ui}h1{font-size:27px}p{color:#c5d0dc}a{color:#afd7ff}select,button{font:inherit;color:white;background:#283c52;border:1px solid #68839f;border-radius:6px;padding:8px;margin:4px}video{width:100%;max-height:72vh;background:black}.row{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.scroll{overflow-x:auto}.scroll img{width:1600px;max-width:none}video.expanded{position:fixed;inset:0;width:100vw;height:100vh;max-height:none;z-index:1000;object-fit:contain}#close{position:fixed;top:8px;right:8px;z-index:1001;display:none}.large #close{display:block}#error{color:#ffb9b9}</style>
<h1>加强材质风格：同一 C8 网络</h1><p>保色加强版继续训练2000步；另保留250步高强度试验版，后者有偏灰偏黄副作用。两者视频直接来自各自权重，没有额外锐化或时间混帧。不同训练步数不作为颜色约束的严格消融；同250步对照已单独保存。</p>
<p>走廊12秒 / 台阶12秒 / 广场人物15秒。1080→720→1080。可在同一时刻切换版本并放大观看。</p>
<div class="row"><label>场景<select id="scene" aria-label="场景"><option value="stairs">台阶 / 12秒</option><option value="new_hallway">走廊 / 12秒</option><option value="plaza">广场人物 / 15秒</option></select></label><label>版本<select id="version" aria-label="版本"><option value="new">保色加强版 / 2000步</option><option value="aggressive">高强度试验版 / 250步，有偏色</option><option value="previous">上一轮 C8</option><option value="teacher">教师目标</option><option value="source">原片</option></select></label><button id="play">播放 / 暂停</button><button id="seek">定位3秒</button><button id="expand">放大观看</button></div>
<div id="label"></div><video controls playsinline loop preload="metadata"></video><button id="close">返回对照</button><div id="error"></div>
<h2>原尺寸局部</h2><p>各列：原片 / 教师 / 上一轮 / 本轮 / 加强后的训练目标。可横向滚动。</p>
PANELS
<p><a href="RESULTS.md">实验结果与限制</a> · <a href="final/fullres_c8_fp32.pt">保色版权重 / 2000步</a> · <a href="../fullres-c8-stylepush-20261003/final/fullres_c8_fp32.pt">高强度试验版权重 / 250步</a> · <a href="video_diagnostics.json">连续帧测量</a></p>
<script>const v=document.querySelector('video'),scene=document.querySelector('#scene'),version=document.querySelector('#version'),seek=document.querySelector('#seek');
function load(keep){const t=keep?v.currentTime:0,playing=!v.paused,id=scene.value,k=version.value;v.pause();seek.disabled=true;document.querySelector('#error').textContent='';const base=k==='new'?'videos/':k==='aggressive'?'../fullres-c8-stylepush-20261003/videos/':k==='previous'?'../fullres-c8-multiscene-20261003/videos/':k==='teacher'?'../b1-distillation-20261003/videos-2000/':'../clear-style-control-20261003/video_sources/';const suffix=['new','previous','aggressive'].includes(k)?'c8':k;document.querySelector('#label').textContent=scene.options[scene.selectedIndex].text+' / '+version.options[version.selectedIndex].text;v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration-.1);seek.disabled=false;if(playing)v.play().catch(()=>{})};v.src=base+id+'_'+suffix+'_1080p.mp4'}
scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();seek.onclick=()=>{v.pause();v.currentTime=3};v.onerror=()=>document.querySelector('#error').textContent='视频读取失败';function large(on){v.classList.toggle('expanded',on);document.body.classList.toggle('large',on)}document.querySelector('#expand').onclick=()=>large(true);document.querySelector('#close').onclick=()=>large(false);document.addEventListener('keydown',e=>{if(e.key==='Escape')large(false)});load(false);</script></html>'''
    panels='\n'.join(f'<div class="scroll"><a href="train/validation_002000/{name}_comparison.png"><img src="train/validation_002000/{name}_comparison.png" alt="{name}原尺寸对照"></a></div>' for name in ('val_stairs_090','val_plaza_090','val_new_hallway_120'))
    (run/'index.html').write_text(html.replace('PANELS',panels),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','render','diagnostics','page']);args=ap.parse_args()
    root=configure_workspace();contract_guard();run=root/'runs'/RUN;torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[args.mode](root,run)


if __name__=='__main__':main()
