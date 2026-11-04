"""Reload verified colored weights, render actual 12-second continuous scenes."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .stroke_color_net import StrokeColorNet
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_region_classes import encode,motion_map
from .render_wavelet_game import close_pipe,video_check


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--clips',nargs='+',default=['stairs','plaza','new_hallway']);a=parser.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    run=a.run;state=torch.load(run/'generator_fp32.pt',map_location='cpu',weights_only=True)
    assert state['folded'] and all(file_hash(p)==h for p,h in state['binding']['code'].items())
    cls=StrokeColorNet
    if state['config'].get('graph')=='native-stroke-color-input-v2':
        from .stroke_color_input_net import StrokeColorInputNet
        cls=StrokeColorInputNet
    data=json.loads((run/'data.json').read_text(encoding='utf-8'));net=cls(False).cuda().eval();net.load_state_dict(state['model'])
    ref=cls(True).cuda().eval();ref.load_state_dict(torch.load(run/f'checkpoint_{state["step"]:06d}.pt',map_location='cpu',weights_only=True)['model'])
    out=run/'delivery';out.mkdir(exist_ok=False);checks=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda();y=net(x);err=float((y-ref(x)).abs().max());assert err<=1e-6
        actual=to_rgb(y[0]);expected=read_image(str(run/f'eval-{state["step"]:06d}'/(r['id']+'.png')))
        pe=int(np.abs(actual.astype(int)-expected.astype(int)).max());assert pe==0
        checks.append(dict(id=r['id'],rgb_float_max_error=err,uint8_max_error=pe))
    del ref
    rows=[json.loads(r) for r in (run/'metrics.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(1,state['step']+1))
    write_json(out/'weight_verification.json',dict(status='PASS',checks=checks,sha256=file_hash(run/'generator_fp32.pt'),
        updates=state['step'],native_updates=sum(r['native'] for r in rows),optimizer_seconds=sum(r['seconds'] for r in rows),
        graph=net.budget(),board_validated=False))
    shutil.copy2(run/f'eval-{state["step"]:06d}/comparison.png',out/'comparison.png')
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    teachers=json.loads((root/'runs/b1-distillation-20261003/videos-2000/report.json').read_text(encoding='utf-8'))['records']
    records=[];pages=[]
    for name in a.clips:
        clip=next(c for c in clips if c['id']==name);teacher=next(r for r in teachers if r['id']==name)
        assert clip['source'] not in data['training_sources'] and file_hash(clip['clip'])==clip['clip_sha256']
        assert teacher['clip_sha256']==clip['clip_sha256'] and file_hash(teacher['teacher'])==teacher['teacher_validation']['sha256']
        cap=cv2.VideoCapture(clip['clip']);tc=cv2.VideoCapture(teacher['teacher']);path=out/(name+'_12s_1080p.mp4');pipe=encode(path)
        n=0;duplicates=0;previous=None;motion=[];frame_checks=[];started=time.perf_counter()
        try:
            while True:
                ok,bgr=cap.read();tk,tbgr=tc.read()
                if not ok:assert not tk;break
                assert tk
                source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));target=area_3to2(cv2.cvtColor(tbgr,cv2.COLOR_BGR2RGB))
                y=to_rgb(net(as_tensor(source)[None].cuda())[0]);pipe.stdin.write(bilinear_2to3(y).tobytes())
                if n in (90,120,210):
                    Image.fromarray(y).save(out/f'{name}_{n:03d}_native720.png')
                    match=next((r for r in data['validation'] if r['id']==f'val_{name}_{n:03d}'),None)
                    if match:
                        source_error=int(np.abs(source.astype(int)-read_image(match['source']).astype(int)).max())
                        expected=read_image(str(run/f'eval-{state["step"]:06d}'/(match['id']+'.png')))
                        output_error=int(np.abs(y.astype(int)-expected.astype(int)).max())
                        assert source_error==0 and output_error==0
                        frame_checks.append(dict(id=match['id'],source_uint8_max_error=source_error,output_uint8_max_error=output_error))
                if previous is not None:
                    mx,my,valid=motion_map(previous[0],source);row=dict(frame=n,valid_fraction=float(valid.mean()))
                    for label,before,now in zip(('source','teacher','student'),previous,(source,target,y)):
                        warped=cv2.remap(before.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        row[label+'_warp_mae']=float(np.abs(now.astype(np.float32)-warped)[valid].mean())
                    motion.append(row);duplicates+=int(np.array_equal(y,previous[2]))
                previous=(source,target,y);n+=1
                if n%60==0:print('COLOR_VIDEO',name,n,round(time.perf_counter()-started,1),flush=True)
        finally:cap.release();tc.release();close_pipe(pipe)
        assert n==clip['validation']['frames'];check=video_check(path,n)
        record=dict(id=name,check=check,frames=n,duration=n/30,duplicates=duplicates,source=clip,teacher=teacher['teacher'],
            means={k:float(np.mean([r[k] for r in motion])) for k in ('source_warp_mae','teacher_warp_mae','student_warp_mae')},
            motion=motion,matched_validation_frames=frame_checks,temporal_acceptance=False,style_acceptance=False,pipeline='integer area1080to720 -> actual FP32 model -> integer bilinear720to1080',
            post_filter=False,temporal_blend=False)
        records.append(record);write_json(out/(name+'_video.json'),record)
        pages.append(dict(id=name,student=path.name,source='/'+Path(clip['clip']).relative_to(root/'runs').as_posix(),teacher='/'+Path(teacher['teacher']).relative_to(root/'runs').as_posix()))
    write_json(out/'report.json',dict(records=records,weights_sha256=file_hash(run/'generator_fp32.pt'),style_accepted=False,board_validated=False))
    (out/'INFERENCE.md').write_text('''# 彩色教师效果实验：复现说明

使用项目现有隔离环境 env-probes/animegan/Scripts/python.exe。模型定义为
`network_ab.training.stroke_color_input_net.StrokeColorInputNet(difference=False)`，
载入父目录 generator_fp32.pt 的 model 字段，再设 eval()。输入 NCHW FP32，
每帧 RGB uint8 按 (x-128)/128 归一化，原生1280×720；输出 RGB 三通道，
按 round(y*128+128) 裁剪到0..255。1080输入先整数area到720，输出再整数bilinear到1080。

推理只使用原片和生成器权重。教师、光流、局部纹理判别器和教师描线提取只用于训练或诊断，
没有在实际输出后做模糊、锐化、原片混合或视频混帧。

source目录保存绑定源码；config/data/binding记录超参数、数据与代码哈希。
本次checkpoint可用于续训。FP32 PyTorch文件含元数据，文件体积不等于INT8风格包体积。

当前尚无部署验收：虽然参数数和理想MAC在B1名义上限内，现有固定bias/multiplier
分区各超160字节，并含待适配的1和3输入项点卷积；需重排/折叠、量化、feature分配、
命令生成、RTL、资源和时序验证。本文件不是刷板说明。
''',encoding='utf-8')
    names=dict(stairs='台阶与草丛',plaza='广场与人物',new_hallway='走廊')
    buttons=''.join('<button onclick="scene('+str(i)+')">'+names[p['id']]+'</button>' for i,p in enumerate(pages))
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>轮廓＋颜色 · 教师效果实验</title>
<style>body{background:#161e29;color:#eaf0f6;font:17px system-ui;max-width:1340px;margin:24px auto;padding:0 18px}h1{font-size:26px}p{line-height:1.7}button{background:#283b50;color:white;border:1px solid #6e889c;border-radius:6px;padding:12px;margin:5px;cursor:pointer}video,img{max-width:100%;width:1280px}a{color:#9ee8e8}.note{color:#ffd993}</style>
<h1>轮廓＋颜色：真实教师效果实验</h1><p>单看本次、教师或原片，可在同一时刻切换。每段12秒、30fps、1920×1080，实际链路1080→720网络→1080；没有后置锐化或混帧。</p>
<p class="note">这是实验结果，还不是教师效果或FPGA部署验收。右侧同期轮廓图用于检查轮廓支路是否保留。</p>
SCENES<br><button onclick="view('student')">本次彩色网络</button><button onclick="view('teacher')">教师效果</button><button onclick="view('source')">原片</button><button onclick="v.currentTime=0;v.play()">从头播放12秒</button><button onclick="v.pause();v.currentTime=3">暂停在3秒</button><span id="label">本次彩色网络</span>
<video id="v" controls preload="metadata" src="FIRST"></video><h2>原像素局部对照</h2><img src="comparison.png" alt="原片、教师、本次彩色、同期轮廓对照">
<p><a href="../generator_fp32.pt">真实FP32权重</a> · <a href="weight_verification.json">权重验证</a> · <a href="report.json">连续视频检查</a> · <a href="/style-library/black-white-line-v1-20261004/preview.html">已独立封存：黑白线条风</a></p>
<script>const scenes=DATA;let index=0,kind='student';function swap(){const time=v.currentTime,playing=!v.paused;v.src=scenes[index][kind];label.textContent=({student:'本次彩色网络',teacher:'教师效果',source:'原片'})[kind];v.onloadedmetadata=()=>{v.currentTime=Math.min(time,v.duration);if(playing)v.play()};v.load()}function view(k){kind=k;swap()}function scene(i){index=i;swap()}</script></html>'''
    html=html.replace('SCENES',buttons).replace('FIRST',pages[0]['student']).replace('DATA',json.dumps(pages,ensure_ascii=False))
    (out/'preview.html').write_text(html,encoding='utf-8');print('COLOR_DELIVERY_DONE',out,flush=True)


if __name__=='__main__':main()
