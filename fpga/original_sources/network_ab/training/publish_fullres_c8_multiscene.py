"""Full-resolution video and measured diagnostics for the actual C8 weights."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from .train_fullres_c8_multiscene import RUN
from .probe_fullres_c8 import FullresC8
from .train_b1_distillation import as_tensor,to_rgb
from .fpga_video_resize import area_3to2,bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe,video_check
from .runtime import configure_workspace,contract_guard,file_hash,write_json


def load(run):
    path=run/'final/fullres_c8_fp32.pt';export=json.loads((run/'final/export.json').read_text())
    assert file_hash(path)==export['sha256']
    model=FullresC8().cuda().eval();model.load_state_dict(torch.load(path,weights_only=True)['model']);return model


def verify(root,run):
    cfg=json.loads((run/'config.json').read_text());binding=json.loads((run/'binding.json').read_text())
    for p,digest in binding['code'].items():assert file_hash(p)==digest,p
    rows=[json.loads(line) for line in (run/'train/metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in rows]==list(range(1,2001))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 for r in rows)
    sources=Counter(s['source'] for row in rows for s in row['samples']);assert len(sources)==7
    launch=json.loads((run/'train/launch_000500.json').read_text());assert launch['initial_step']==500
    model=load(run);fused=copy.deepcopy(model)
    for seq in fused.body:
        seq[1]=torch.nn.utils.fuse_conv_bn_eval(seq[1],seq[2]);seq[2]=nn.Identity()
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));errors=[];panels=[]
    with torch.inference_mode():
        for r in manifest['validation']:
            a=np.array(Image.open(r['source']));t=np.array(Image.open(r['target']))
            x=as_tensor(a)[None].cuda();p=model(x);error=float((p-fused(x)).abs().max());assert error<5e-5;errors.append(error)
            native=to_rgb(p[0]);saved=np.array(Image.open(run/'train/validation_002000'/(r['id']+'.png')))
            assert np.max(np.abs(native.astype(int)-saved.astype(int)))<=1
            if r['id'] not in ('val_stairs_090','val_plaza_090','val_new_hallway_120'):continue
            old=np.array(Image.open(root/'runs/b1-distillation-20261003/train/validation_002000'/(r['id']+'.png')))
            crop=(424,320,320,320) if r['id']=='val_stairs_090' else (80,400,320,320) if r['id']=='val_plaza_090' else (560,150,320,320)
            xx,yy,w,h=crop;canvas=Image.new('RGB',(1280,348),(25,25,25));draw=ImageDraw.Draw(canvas)
            for i,(image,label) in enumerate([(a,'Input'),(t,'Teacher'),(old,'B1 / 2000'),(native,'C8 / 2000, 7 sources')]):
                canvas.paste(Image.fromarray(image[yy:yy+h,xx:xx+w]),(i*320,28));draw.text((i*320+8,8),label,fill='white')
            name=r['id']+'_comparison_1to1.png';canvas.save(run/name);panels.append(name)
    info=dict(status='FP32_CHECKS_PASS_VISUAL_ASSESSMENT_REQUIRED',steps=len(rows),source_samples=dict(sources),
              native_steps=sum(r['native'] for r in rows),optimizer_seconds=sum(r['seconds'] for r in rows),
              peak_mib=max(r['peak_mib'] for r in rows),resume_process_at=500,
              preflight=json.loads((run/'preflight/result.json').read_text())['replay_error'],
              bn_fold_max_error=max(errors),comparison_panels=panels,weights_sha256=file_hash(run/'final/fullres_c8_fp32.pt'),
              production=False,qat=False,hardware_modified=False)
    write_json(run/'verification.json',info);print(json.dumps(info),flush=True)


def render(root,run):
    model=load(run);out=run/'videos';out.mkdir(exist_ok=False)
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    records=[]
    for clip in clips:
        assert file_hash(clip['clip'])==clip['clip_sha256']
        assert 10<=clip['validation']['seconds']<=20
        name=clip['id'];path=out/(name+'_c8_1080p.mp4')
        cap=cv2.VideoCapture(clip['clip']);pipe=encode_pipe(path,1920,1080,30);n=0;begin=time.perf_counter()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert bgr.shape==(1080,1920,3)
            small=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            with torch.inference_mode():result=to_rgb(model(as_tensor(small)[None].cuda())[0])
            shown=bilinear_2to3(result);pipe.stdin.write(shown.tobytes())
            if n in (0,90,120,210,359,449):
                Image.fromarray(result).save(out/f'{name}_{n:03d}_native720.png')
                Image.fromarray(shown).save(out/f'{name}_{n:03d}_preencode1080.png')
            n+=1
            if n%90==0:print('C8_VIDEO',name,n,round(time.perf_counter()-begin,1),flush=True)
        cap.release();close_pipe(pipe);assert n==clip['validation']['frames']
        records.append(dict(id=name,path=str(path),source=clip['clip'],source_sha256=clip['clip_sha256'],
                            validation=video_check(path,n),model_calls=n,seconds=time.perf_counter()-begin))
        write_json(out/'report.json',dict(status='COMPLETE' if len(records)==len(clips) else 'RENDERING',records=records,
             pipeline='1920x1080 -> integer area1280x720 -> C8 -> integer bilinear1920x1080',
             postprocess=False,source_blend=False,temporal_blend=False,frame_repeat=False,crf=16))
    print('C8_VIDEOS_COMPLETE',flush=True)


def diagnose(root,run):
    report=json.loads((run/'videos/report.json').read_text());assert report['status']=='COMPLETE';results=[]
    old_report=json.loads((root/'runs/b1-distillation-20261003/videos-2000/report.json').read_text())
    # All variants decoded from the actual delivered videos, on the same 720 grid.
    # Source-derived optical-flow validity is shared by every method.
    for clip in report['records']:
        name=clip['id'];paths=dict(source=clip['source'],c8=clip['path'],
          b1=str(root/f'runs/b1-distillation-20261003/videos-2000/{name}_b1_1080p.mp4'),
          teacher=str(root/f'runs/b1-distillation-20261003/videos-2000/{name}_teacher_1080p.mp4'))
        old=next(row for row in old_report['records'] if row['id']==name)
        assert old['clip_sha256']==clip['source_sha256']==file_hash(paths['source'])
        assert file_hash(paths['c8'])==clip['validation']['sha256']
        assert file_hash(paths['b1'])==old['student_validation']['sha256']
        assert file_hash(paths['teacher'])==old['teacher_validation']['sha256']
        caps={k:cv2.VideoCapture(v) for k,v in paths.items()};prev=None;frames=0
        measures={k:[] for k in ('c8','b1','teacher')};coverage=[];highlights=Counter()
        comparisons={k:[] for k in ('c8','b1')};changes={k:[] for k in ('c8','b1','teacher')}
        yy,xx=np.mgrid[:720,:1280].astype(np.float32)
        while True:
            raw={k:c.read() for k,c in caps.items()}
            if not raw['source'][0]:assert not any(v[0] for v in raw.values());break
            assert all(v[0] for v in raw.values())
            rgb={k:area_3to2(cv2.cvtColor(v[1],cv2.COLOR_BGR2RGB)) for k,v in raw.items()}
            gray={k:cv2.cvtColor(v,cv2.COLOR_RGB2GRAY).astype(np.float32) for k,v in rgb.items()}
            # Full 720-grid temporal measurement, flow estimated at half resolution.
            if prev is not None:
                cur=cv2.resize(gray['source'],(640,360));past=cv2.resize(prev['source'],(640,360))
                back=cv2.calcOpticalFlowFarneback(cur,past,None,.5,3,21,3,5,1.2,0)
                forward=cv2.calcOpticalFlowFarneback(past,cur,None,.5,3,21,3,5,1.2,0)
                flow=cv2.resize(back,(1280,720))*2;fw=cv2.resize(forward,(1280,720))*2
                mx,my=xx+flow[:,:,0],yy+flow[:,:,1]
                sample=lambda a:cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                src_err=gray['source']-sample(prev['source'])
                consistency=np.linalg.norm(flow+sample(fw),axis=2)
                valid=(np.abs(src_err)<8)&(consistency<1.5)&(mx>=1)&(mx<1278)&(my>=1)&(my<718)
                coverage.append(float(valid.mean()))
                for key in measures:
                    delta=(gray[key]-gray['source'])-sample(prev[key]-prev['source'])
                    measures[key].append(float(np.abs(delta[valid]).mean()) if valid.any() else None)
            for key in changes:
                changes[key].append(float(np.abs(rgb[key].astype(np.float32)-rgb['source']).mean()))
                highlights[key]+=int(((rgb[key].min(2)>245)&(rgb['source'].max(2)<220)).sum())
            for key in comparisons:comparisons[key].append(float(np.abs(rgb[key].astype(np.float32)-rgb['teacher']).mean()))
            prev=gray;frames+=1
            if frames%120==0:print('VIDEO_DIAGNOSTIC',name,frames,flush=True)
        for c in caps.values():c.release()
        assert frames==clip['validation']['frames']
        summary={}
        for key,values in measures.items():
            v=np.array([x for x in values if x is not None]);assert len(v)==frames-1
            summary[key]=dict(flow_aligned_effect_change_luma_mean=float(v.mean()),
                frame_error_p95=float(np.percentile(v,95)),source_change_rgb_mean=float(np.mean(changes[key])),
                newly_white_pixel_fraction=highlights[key]/(frames*720*1280))
            if key in comparisons:summary[key]['teacher_mae_rgb_mean']=float(np.mean(comparisons[key]))
        results.append(dict(id=name,frames=frames,variants=summary,flow_valid_mean=float(np.mean(coverage)),
                            input_hashes={k:file_hash(v) for k,v in paths.items()}))
        write_json(run/'video_diagnostics.json',dict(status='MEASURED_NOT_VISUAL_PASS',records=results,
          scope='All delivered codecs decoded to common area720 grid; source-derived flow and mask shared across methods.',
          limitations=['A smaller temporal error can result from a weaker style; this is not a stand-alone quality score.',
                       'Flow can be inaccurate around occlusion/thin plants. Human video review is still required.',
                       'New-white test excludes source-bright pixels; it is not a universal exposure metric.']))
    print('DIAGNOSTICS_COMPLETE',flush=True)


def page(run):
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C8多场景训练：连续视频对照</title>
<style>body{max-width:1280px;margin:24px auto;padding:0 18px;background:#171c23;color:#edf3f7;font:16px/1.65 system-ui}h1{font-size:27px}p{color:#c5d0dc}a{color:#afd7ff}.note{padding:14px;background:#29313b;border-left:4px solid #d9b76c}select,button{font:inherit;color:white;background:#283c52;border:1px solid #68839f;border-radius:6px;padding:8px;margin:4px}video{width:100%;max-height:72vh;background:black}.row{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.scroll{overflow-x:auto}.scroll img{width:1280px;max-width:none;display:block}#error{color:#ffc0c0}</style>
<h1>小网络多场景训练：2000步</h1>
<p class="note">从新初始化训练，7个来源、95组相邻帧。这里的走廊、台阶、广场均未进入本轮训练，但之前已用于查看与诊断，因此不称盲测。网络输出没有锐化、原图混合或时间平滑。尚未完成FPGA部署。</p>
<h2>连续视频</h2><p>走廊12秒、台阶12秒、广场15秒；1080→720处理→1080输出。同一场景切换版本会保留播放位置。</p>
<div class="row"><label>场景<select id="scene" aria-label="场景"><option value="stairs">台阶 / 12秒</option><option value="new_hallway">走廊 / 12秒</option><option value="plaza">广场人物 / 15秒</option></select></label><label>版本<select id="version" aria-label="版本"><option value="c8">本轮 C8</option><option value="teacher">教师目标</option><option value="b1">之前 B1</option><option value="source">原片</option></select></label><button id="play">播放 / 暂停</button><button id="seek">定位3秒</button></div>
<div id="label"></div><video controls playsinline loop preload="metadata"></video><div id="error"></div>
<h2>原尺寸局部：原图 / 教师 / B1 / 本轮</h2><p>每列320×320；窄窗口可横向滚动查看，避免缩图掩盖模糊。</p>
<div class="scroll"><a href="val_stairs_090_comparison_1to1.png"><img src="val_stairs_090_comparison_1to1.png" alt="台阶原尺寸对照"></a></div>
<div class="scroll"><a href="val_plaza_090_comparison_1to1.png"><img src="val_plaza_090_comparison_1to1.png" alt="人物原尺寸对照"></a></div>
<div class="scroll"><a href="val_new_hallway_120_comparison_1to1.png"><img src="val_new_hallway_120_comparison_1to1.png" alt="走廊原尺寸对照"></a></div>
<p><a href="RESULTS.md">实验结论</a> · <a href="verification.json">训练与重载验证</a> · <a href="video_diagnostics.json">视频测量及限制</a> · <a href="final/fullres_c8_fp32.pt">本轮真实FP32权重</a></p>
<script>const v=document.querySelector('video'),scene=document.querySelector('#scene'),version=document.querySelector('#version'),seek=document.querySelector('#seek');
function load(keep){const t=keep?v.currentTime:0,playing=!v.paused,id=scene.value,k=version.value;v.pause();seek.disabled=true;document.querySelector('#error').textContent='';let path=k==='c8'?'videos/'+id+'_c8_1080p.mp4':k==='source'?'../clear-style-control-20261003/video_sources/'+id+'_source_1080p.mp4':'../b1-distillation-20261003/videos-2000/'+id+'_'+k+'_1080p.mp4';document.querySelector('#label').textContent=scene.options[scene.selectedIndex].text+' / '+version.options[version.selectedIndex].text;v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration-.1);seek.disabled=false;if(playing)v.play().catch(()=>{})};v.src=path;}
scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();seek.onclick=()=>{v.pause();v.currentTime=3};v.onerror=()=>document.querySelector('#error').textContent='视频读取失败';load(false);</script></html>'''
    (run/'index.html').write_text(html,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','render','diagnose','page']);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='page':page(run)
    else:globals()[args.mode](root,run)


if __name__=='__main__':main()
