"""Reviewable seven-level causal pipeline, explicitly including fixed fill code."""
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .contour_guided_fill import contours,fill
from .train_seven_contours import LEVELS
from .stroke_color_input_net import StrokeColorInputNet
from .train_native_contours import ROOT,tensor,load_rgb
from .train_b1_distillation import to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_moderate_contours import encode
from .render_wavelet_game import close_pipe,video_check
from .publish_region_classes import motion_map

STRENGTH=.12
ITERATIONS=80
RUN_NAME='seven-structured-fill-v1'
PIPELINE_EXTRA={}
PAGE_EXTRA=''


def error_stats(a,b):
    d=np.abs(a.astype(np.float32)-b.astype(np.float32))
    return dict(rgb255_mae=float(d.mean()),maximum=float(d.max()))


@torch.no_grad()
def main():
    w=configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    out=ROOT/RUN_NAME/'delivery';out.mkdir(exist_ok=False)
    old=ROOT/'native-stroke-color-texture-v4';state=torch.load(old/'generator_fp32.pt',map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model'])
    shutil.copy2(old/'generator_fp32.pt',out/'student_RGB_fp32.pt')
    data=json.loads((old/'data.json').read_text());shutil.copy2(old/'data.json',out/'data.json')
    config=dict(levels=LEVELS,strength=STRENGTH,iterations=ITERATIONS,
        sequence=['original RGB -> actual learned student RGB field','student RGB field -> fixed seven-level contour decoder',
                  'student RGB field AND decoded contours -> boundary-weighted region-color simplification'],
        extra_inference_operators=['Gaussian','Canny NMS/hysteresis','connected-component selection','80-step weighted TV fill'],
        not_weights_only=True,board_validated=False,new_neural_training=False,
        source_weight=str(old/'generator_fp32.pt'),source_weight_sha256=file_hash(old/'generator_fp32.pt'),
        no_teacher_at_inference=True,no_output_temporal_blend=True)
    config.update(PIPELINE_EXTRA)
    write_json(out/'pipeline.json',config);records=[]
    for row in data['validation']:
        source=load_rgb(row['source']);pred=to_rgb(net(tensor(source)[None].cuda())[0]);expected=load_rgb(old/'eval-001600'/(row['id']+'.png'))
        assert np.array_equal(pred,expected)
        Image.fromarray(source).save(out/(row['id']+'_source.png'));Image.fromarray(pred).save(out/(row['id']+'_before.png'))
        shutil.copy2(row['target'],out/(row['id']+'_teacher.png'))
        for l in LEVELS:
            e=contours(pred,l);rgb=fill(pred,e,STRENGTH,ITERATIONS)
            Image.fromarray(np.uint8((1-e)*255)).save(out/(row['id']+'_'+l+'_contour.png'))
            Image.fromarray(rgb).save(out/(row['id']+'_'+l+'_color.png'))
            records.append(dict(id=row['id'],level=l,contour_density=float(e.mean()),change_from_old=error_stats(rgb,pred)))
        if row['id']=='val_stairs_090':
            e=contours(pred,'g4');actual=fill(pred,e,STRENGTH,ITERATIONS)
            shifted=fill(pred,np.roll(e,(47,83),(0,1)),STRENGTH,ITERATIONS)
            noedge=fill(pred,np.zeros_like(e),STRENGTH,ITERATIONS)
            Image.fromarray(shifted).save(out/'shifted_contour_fill.png');Image.fromarray(noedge).save(out/'no_contour_fill.png')
            causal=dict(displaced_contour=error_stats(actual,shifted),zero_contour=error_stats(actual,noedge))
        print('SEVEN_STILLS',row['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    board=Image.new('RGB',(1600,948),'#17202a');draw=ImageDraw.Draw(board)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    for j,(key,box) in enumerate(cases):
        for i,(suffix,title) in enumerate([('teacher','教师参考'),('before','修正前实际输出'),('g4_contour','本次实际使用的中档轮廓'),('g4_color','轮廓参与颜色合并后'),('source','原片')]):
            draw.text((320*i+4,316*j+4),title,font=font,fill='white');board.paste(Image.open(out/(key+'_'+suffix+'.png')).convert('RGB').crop(box),(320*i,316*j+34))
    board.save(out/'comparison.png');write_json(out/'stills_verification.json',dict(records=records,causal_fill_test=causal,reloaded_RGB_matches_all6=True))
    clips=json.loads((w/'runs/clear-style-control-20261003/video_clips.json').read_text())['clips'];clip=next(c for c in clips if c['id']=='stairs')
    assert file_hash(clip['clip'])==clip['clip_sha256'] and clip['source'] not in data['training_sources']
    cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);pipe=encode(out/'middle_color_12s_1080p.mp4')
    n=0;previous=None;motion=[];dups=0;start=time.perf_counter()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pred=to_rgb(net(tensor(source)[None].cuda())[0]);e=contours(pred,'g4');rgb=fill(pred,e,STRENGTH,ITERATIONS)
            pipe.stdin.write(bilinear_2to3(rgb).tobytes())
            if n in (90,210):
                assert np.array_equal(rgb,load_rgb(out/f'val_stairs_{n:03d}_g4_color.png'))
                assert np.array_equal(source,load_rgb(out/f'val_stairs_{n:03d}_source.png'))
            if previous is not None:
                mx,my,valid=motion_map(previous[0],source);rec=dict(frame=n,valid_fraction=float(valid.mean()))
                for key,a,b in [('original',previous[0],source),('before',previous[1],pred),('after',previous[2],rgb)]:
                    warped=cv2.remap(a.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    rec[key]=float(np.abs(b.astype(np.float32)-warped)[valid].mean())
                motion.append(rec);dups+=int(np.array_equal(rgb,previous[2]))
            previous=(source,pred,rgb);n+=1
            if n%30==0:print('FILLED_VIDEO',n,round(time.perf_counter()-start,1),flush=True)
    finally:
        cap.release();close_pipe(pipe)
    assert n==360
    verified=video_check(out/'middle_color_12s_1080p.mp4',n)
    write_json(out/'video_verification.json',dict(video=verified,motion=motion,motion_mean={k:float(np.mean([r[k] for r in motion])) for k in ('original','before','after')},
        source=clip,duplicate_frames=dups,stills_match_native_frames=[90,210],pipeline='1080 integer area -> native720 complete experimental pipeline -> integer bilinear1080',
        temporal_acceptance=False,style_acceptance=False,board_validated=False))
    files=['contour_guided_fill.py','coarse_contour_targets.py','train_seven_contours.py','stroke_color_input_net.py','stroke_color_net.py','native_contour_net.py','train_context_contours.py','fpga_video_resize.py',Path(__file__).name]
    files+=PIPELINE_EXTRA.get('additional_source_files',[])
    (out/'source').mkdir()
    for name in files:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>七档轮廓与实际颜色合并</title><style>body{background:#17202b;color:#edf3f8;font:17px system-ui;margin:24px auto;max-width:1560px;padding:0 20px}p{line-height:1.7}button,select{font:inherit;padding:9px;margin:5px;background:#30455c;color:white;border:1px solid #b2c8d9}img,video{max-width:100%;display:block}.warning{color:#ffe0a6}a{color:#9edcf7}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}</style>
<h1>轮廓真正进入颜色合并 · 七档实验</h1><p>旧代码的轮廓只作为弱提示，原片特征仍进入RGB输出。下面改为：实际学生预测 → 提取所选档位轮廓 → 用轮廓约束块内颜色合并。</p>
<p class="warning">这是完整实验处理链，不是“网络本身已经学会”或“只换权重即可上板”。含明确的轮廓解码和颜色合并算法，尚未通过FPGA资源、时序与部署验收。</p>
<select id="scene" onchange="show()"><option value="val_stairs_090">台阶与草丛</option><option value="val_stairs_210">台阶后段</option><option value="val_plaza_090">人物与树木</option><option value="val_plaza_210">广场后段</option><option value="val_new_hallway_120">走廊</option><option value="val_new_hallway_210">走廊后段</option></select>
<span id="buttons"></span><p id="label">4档（中间档）</p><div class="row"><div>实际参与上色的轮廓<img id="contour" alt="实际使用的轮廓"></div><div>经过轮廓约束的实际颜色结果<img id="color" alt="实际颜色结果"></div></div>
<p><button onclick="side('before')">看修正前RGB输出</button><button onclick="side('source')">看原片</button><button onclick="side('teacher')">看教师参考</button><button onclick="show()">返回当前档结果</button></p>
<h2>中间档真实12秒视频</h2><button onclick="v.currentTime=0;v.play()">从头播放12秒</button><button onclick="v.pause();v.currentTime=3">暂停看3秒</button><video id="v" controls preload="metadata" src="middle_color_12s_1080p.mp4"></video>
<p>360帧、30fps、1920×1080；1080→720处理→1080。没有重复帧或混帧。视频只对应第4档；上面的静帧可切换全部7档。</p><img src="comparison.png" alt="修正前后与教师实际对照">
<p><a href="student_RGB_fp32.pt">实际使用的学生FP32权重</a> · <a href="pipeline.json">完整附加处理配置</a> · <a href="stills_verification.json">七档与轮廓因果检查</a> · <a href="video_verification.json">视频检查</a> · <a href="../..//contour-color-path-audit-v1/causal_comparison.png">旧代码分支干预对照</a></p>
<script>let grade=4;for(let i=1;i<=7;i++){let b=document.createElement('button');b.textContent=i+'档';b.onclick=()=>{grade=i;show()};buttons.appendChild(b)}function show(){let s=scene.value;contour.src=s+'_g'+grade+'_contour.png';color.src=s+'_g'+grade+'_color.png';label.textContent=grade+'档（1：轻减细节，7：较强减细节）'}function side(k){color.src=scene.value+'_'+k+'.png'}show()</script></html>'''
    page=page.replace('<h2>中间档真实12秒视频</h2>',PAGE_EXTRA+'<h2>中间档真实12秒视频</h2>')
    (out/'preview.html').write_text(page,encoding='utf-8')
    write_json(out/'manifest.json',dict(files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in out.rglob('*') if p.is_file()},pipeline=config))
    print('SEVEN_FILLED_DELIVERY_DONE',out,flush=True)


if __name__=='__main__':main()
