"""Export three independently loadable moderate-grain contour weights/videos."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .train_moderate_contours import GrainContourNet
from .train_context_contours import ContextContourNet
from .moderate_contour_targets import LEVELS
from .train_native_contours import ROOT,tensor,load_rgb
from .fpga_video_resize import area_3to2,bilinear_2to3,_up_axis
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .media import decoder_identity
from .publish_region_classes import motion_map
from .render_wavelet_game import close_pipe,video_check


def encode(path):
    return subprocess.Popen([decoder_identity()['path'],'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s:v','1920x1080','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-threads','1','-crf','16',
        '-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(path)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)


def up_mono(mono):
    # Same integer arithmetic as RGB scaling; identical channels need be scaled only once.
    return np.repeat(_up_axis(_up_axis(mono[:,:,None],1),0),3,axis=2)


@torch.no_grad()
def main():
    workspace=configure_workspace();contract_guard();set_reproducible(2026100481);torch.set_num_threads(4);cv2.setNumThreads(2)
    run=ROOT/'moderate-contour-grains-v1';state=torch.load(run/'contour_fp32.pt',map_location='cpu',weights_only=True)
    assert state['folded'] and all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=GrainContourNet(False).cuda().eval();net.load_state_dict(state['model'])
    data=json.loads((run/'data.json').read_text(encoding='utf-8'));out=run/'delivery';out.mkdir(exist_ok=False)
    rows=[json.loads(v) for v in (run/'metrics.jsonl').read_text().splitlines()]
    assert [v['step'] for v in rows]==list(range(1,state['step']+1))
    checks=[];exported={}
    for k,level in enumerate(LEVELS):
        model={name:value.clone() for name,value in state['model'].items()}
        model['edge.weight']=model['edge.weight'][k:k+1];model['edge.bias']=model['edge.bias'][k:k+1]
        path=out/(level+'_contour_fp32.pt')
        torch.save(dict(model=model,folded=True,level=level,training_step=state['step'],source_sha256=file_hash(run/'contour_fp32.pt'),
            config=state['config'],binding=state['binding'],inference_class='ContextContourNet(difference=False)',
            grayscale_only=True,approved_for_coloring=False,board_validated=False),path)
        single=ContextContourNet(False).cuda().eval();single.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        for r in data['validation']:
            x=tensor(load_rgb(r['source']))[None].cuda();actual=single(x);reference=net(x)[:,k:k+1]
            err=float((actual-reference).abs().max());assert err<=1e-5
            gray=np.uint8((1-actual.sigmoid()[0,0].cpu().numpy())*255)
            previous=np.array(Image.open(run/f'eval-{state["step"]:06d}'/(r['id']+'_'+level+'.png')))
            pe=int(np.abs(gray.astype(int)-previous.astype(int)).max());assert pe<=1
            checks.append(dict(id=r['id'],level=level,split_head_float_error=err,uint8_error=pe))
        exported[level]=dict(path=str(path),sha256=file_hash(path),parameters=sum(p.numel() for p in single.parameters()))
        del single
    shutil.copy2(run/f'eval-{state["step"]:06d}/comparison.png',out/'comparison.png')
    clips=json.loads((workspace/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    clip=next(c for c in clips if c['id']=='stairs')
    assert clip['source'] not in data['training_sources'] and file_hash(clip['clip'])==clip['clip_sha256']
    cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);pipes={l:encode(out/(l+'_12s_1080p.mp4')) for l in LEVELS}
    previous=None;motion=[];duplicates={l:0 for l in LEVELS};n=0;start=time.perf_counter()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));probs=net(tensor(source)[None].cuda()).sigmoid()[0].cpu().numpy()
            mono=np.uint8(np.clip(1-probs,0,1)*255)
            if n==0:
                assert np.array_equal(up_mono(mono[0]),bilinear_2to3(np.repeat(mono[0,:,:,None],3,2)))
            for k,l in enumerate(LEVELS):
                pipes[l].stdin.write(up_mono(mono[k]).tobytes())
                if n in (90,210):
                    Image.fromarray(mono[k]).save(out/f'{l}_{n:03d}_native720.png')
                    expected=np.array(Image.open(run/f'eval-{state["step"]:06d}'/(f'val_stairs_{n:03d}_{l}.png')))
                    assert np.array_equal(mono[k],expected)
            if previous is not None:
                mx,my,valid=motion_map(previous[0],source);record=dict(frame=n,valid_fraction=float(valid.mean()))
                for k,l in enumerate(LEVELS):
                    warped=cv2.remap(previous[1][k],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    band=((probs[k]>.15)|(warped>.15))&valid;error=np.abs(probs[k]-warped)
                    record[l]=dict(probability_mae=float(error[valid].mean()),edge_band_mae=float(error[band].mean()) if band.any() else None)
                    duplicates[l]+=int(np.array_equal(mono[k],previous[2][k]))
                motion.append(record)
            previous=(source,probs,mono);n+=1
            if n%60==0:print('MODERATE_VIDEO',n,round(time.perf_counter()-start,1),flush=True)
    finally:
        cap.release()
        for p in pipes.values():close_pipe(p)
    assert n==360
    videos={l:video_check(out/(l+'_12s_1080p.mp4'),n) for l in LEVELS}
    report=dict(steps=state['step'],contiguous_updates=len(rows),native_updates=sum(r['native'] for r in rows),checks=checks,
        independent_weights=exported,videos=videos,duplicates=duplicates,motion=motion,source=clip,
        pipeline='1080 integer area -> native720 predicted contours -> mathematically identical mono integer bilinear -> RGB1080',
        no_inference_filter=True,no_temporal_blend=True,no_color=True,user_approval_before_coloring=True,board_validated=False,
        single_level_parameters=9529)
    write_json(out/'verification.json',report)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>旧版与A之间 · 三档真实轮廓</title>
<style>body{background:#17202b;color:#eaf1f5;font:17px system-ui;max-width:1600px;margin:24px auto;padding:0 16px}h1{font-size:26px}p{line-height:1.7}button{background:#293e54;color:#fff;border:1px solid #7996b4;border-radius:5px;padding:12px;margin:5px}video{width:min(100%,1280px);display:block}img{max-width:100%}a{color:#9be5ef}.note{color:#ffdb93}</style>
<h1>只减轮廓细节，先不上色</h1><p>三档都在旧版与之前A之间。下面是原片输入后的模型实际预测；没有对预测做模糊、Canny、删线或加粗。</p>
<button onclick="swap('light')">轻减细节</button><button onclick="swap('middle')">中间档</button><button onclick="swap('strong')">较强减细节</button><button onclick="swap('old')">已封存旧版</button><button onclick="v.currentTime=0;v.play()">从头播放12秒</button><button onclick="v.pause();v.currentTime=3">暂停看3秒</button><span id="label">中间档</span>
<video id="v" controls preload="metadata" src="middle_12s_1080p.mp4"></video><p>12秒、360帧、30fps、1920×1080；实际链路1080→720网络→1080。</p>
<img src="comparison.png" alt="教师参考、已封存旧版以及三档真实预测的原像素裁剪对比">
<p class="note">先由你确认轮廓精细程度，再做块内简化和上色。这次不把轮廓多少直接当作效果达标；三档都尚未通过FPGA部署验收。</p>
<p><a href="light_contour_fp32.pt">轻档真实权重</a> · <a href="middle_contour_fp32.pt">中档真实权重</a> · <a href="strong_contour_fp32.pt">较强档真实权重</a> · <a href="verification.json">权重与视频核验</a> · <a href="/style-library/black-white-line-v1-20261004/preview.html">原版黑白线条风归档</a></p>
<script>const paths={light:'light_12s_1080p.mp4',middle:'middle_12s_1080p.mp4',strong:'strong_12s_1080p.mp4',old:'/style-library/black-white-line-v1-20261004/contours_12s_1080p.mp4'},names={light:'轻减细节',middle:'中间档',strong:'较强减细节',old:'已封存旧版'};function swap(k){const t=v.currentTime,playing=!v.paused;v.src=paths[k];label.textContent=names[k];v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration);if(playing)v.play()};v.load()}</script></html>'''
    (out/'preview.html').write_text(html,encoding='utf-8')
    (out/'INFERENCE.md').write_text('三档权重均可独立用 ContextContourNet(difference=False) 载入model字段。输入NCHW FP32 (RGB uint8-128)/128，原生1280x720。输出signed单通道logit；显示按uint8((1-sigmoid(logit))*255)映射为白底黑线。三档由同一轮多头训练得到，每档导出独立9529参数网络，推理不需要其他档或教师。颜色重建已停止，用户认可粒度前不得继续上色。不是刷板包。\n',encoding='utf-8')
    print('MODERATE_DELIVERY_DONE',out,flush=True)


if __name__=='__main__':main()
