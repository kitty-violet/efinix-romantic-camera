"""Publish real model + scale-first fill results, keeping all stages explicit."""
import ast
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .coarse_surface_fill import process, tv_surface, to_tensor, to_array, LEVELS, COLOR
from .contour_guided_fill import gradient, divergence
from .stroke_color_input_net import StrokeColorInputNet
from .train_native_contours import ROOT, tensor, load_rgb
from .train_b1_distillation import to_rgb
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .fpga_video_resize import area_3to2, bilinear_2to3
from .publish_moderate_contours import encode
from .render_wavelet_game import close_pipe, video_check
from .publish_region_classes import motion_map

RUN = ROOT/'coarse-surface-color-v3'
OLD = ROOT/'seven-structured-fill-cool-v2/delivery'


def describe(rgb):
    lab=cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2Lab)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    fine=g-cv2.GaussianBlur(g,(0,0),1)
    return dict(hsv_s_mean=float(hsv[:,:,1].mean()),lab_a=float(lab[:,:,1].mean()),
                lab_b=float(lab[:,:,2].mean()),lab_l=float(lab[:,:,0].mean()),
                fine_detail_rms=float(np.sqrt((fine**2).mean())),
                near_white_fraction=float((rgb.min(2)>=245).mean()),
                flat_neighbor_fraction=float(((np.abs(np.diff(rgb.astype(float),axis=1))).mean(2)<2).mean()))


def edge_info(e):
    p=cv2.dilate(e,np.ones((3,3),np.uint8))
    return dict(edge_fraction=float(e.mean()),protected_fraction=float(p.mean()))


@torch.no_grad()
def checks():
    results={}
    for name in ['coarse_surface_fill.py',Path(__file__).name]:
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    x=torch.ones(1,3,64,96,device='cuda')*.37
    results['constant_max_error']=max(float((tv_surface(x,None,s,0,64)-x).abs().max()) for s in [.025,.055,.085])
    assert results['constant_max_error']<1e-6
    x=torch.randn_like(x);p=torch.randn_like(x);q=torch.randn_like(x)
    p[:,:,:,-1]=0;q[:,:,-1,:]=0;gx,gy=gradient(x)
    results['adjoint_relative_error']=float(((gx*p+gy*q).sum()+(x*divergence(p,q)).sum()).abs()/(gx*p+gy*q).sum().abs().clamp_min(1))
    assert results['adjoint_relative_error']<1e-5
    # A hard step must remain a hard step; this excludes a Gaussian blur substitute.
    x=torch.full((1,3,96,192),.2,device='cuda');x[:,:,:,96:]=.8
    y=tv_surface(x,None,.055,0,64)
    y=tv_surface(y,None,.135,0,96)
    profile=y[0,0,48].cpu().numpy()
    results['step_intermediate_width_pixels']=int(((profile>.3)&(profile<.7)).sum())
    results['step_contrast']=float(profile[144:].mean()-profile[:48].mean())
    assert results['step_intermediate_width_pixels']<=2 and results['step_contrast']>.55
    results['finite']=bool(torch.isfinite(y).all())
    assert results['finite']
    return results


def contact(out):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),
           ('val_plaza_090',(80,400,400,680)),('val_new_hallway_120',(0,0,320,280))]
    b=Image.new('RGB',(1600,1264),'#17202a');d=ImageDraw.Draw(b)
    for j,(key,box) in enumerate(cases):
        for i,(suffix,title) in enumerate([('teacher','教师参考'),('previous','上一版'),('g1_color','本次轻档'),('g4_color','本次中档'),('g7_color','本次较强档')]):
            d.text((320*i+4,316*j+4),title,font=font,fill='white')
            b.paste(Image.open(out/(key+'_'+suffix+'.png')).convert('RGB').crop(box),(320*i,316*j+34))
    b.save(out/'comparison.png')
    b=Image.new('RGB',(1280,2*(360+34)),'#17202a');d=ImageDraw.Draw(b)
    for j,key in enumerate(['val_stairs_090','val_new_hallway_120']):
        for i,(suffix,title) in enumerate([('previous','上一版'),('g4_color','本次中档')]):
            d.text((i*640+8,j*394+4),title,font=font,fill='white')
            b.paste(Image.open(out/(key+'_'+suffix+'.png')).resize((640,360)),(i*640,j*394+34))
    b.save(out/'overview.png')


def page(out):
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>粗色块与冷色修正版</title>
<style>body{background:#17202b;color:#edf3f8;font:17px system-ui;margin:24px auto;max-width:1560px;padding:0 20px}p{line-height:1.7}button,select{font:inherit;padding:9px;margin:5px;background:#30455c;color:white;border:1px solid #b2c8d9}img,video{max-width:100%;display:block}a{color:#9edcf7}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.note{color:#ffe0a6}</style>
<h1>粗色块 · 冷色与强光修正版</h1><p>先去掉细小纹理，再用这一表面生成较疏轮廓并合并块内颜色。草丛不再把每个碎纹都当成必须保留的边界。轻微加强冷色和彩度，调整明暗对比。默认4档，1—7档逐步变粗。</p>
<p class="note">真实学生网络预测 + 新的固定颜色合并处理。神经网络权重沿用上一版；本页效果尚不是仅更换B1权重即可上板的效果。</p>
<h2>12秒完整视频 · 1080→720→1080</h2><select id="movie" onchange="changeMovie()"><option value="stairs">台阶与草丛</option><option value="new_hallway">强光走廊</option></select>
<button id="play" onclick="v.currentTime=0;v.play()">从头播放12秒</button><button onclick="v.pause();v.currentTime=3">暂停看3秒</button>
<video id="v" controls preload="metadata" src="stairs_middle_12s_1080p.mp4"></video><p>每段360帧、30fps、1920×1080，视频对应4档。没有重复帧补时，也没有前后帧混合。</p>
<h2>原生720处理结果 · 七档比较</h2><select id="scene" onchange="show()"><option value="val_stairs_090">台阶与草丛</option><option value="val_stairs_210">台阶后段</option><option value="val_plaza_090">人物与树木</option><option value="val_plaza_210">广场后段</option><option value="val_new_hallway_120">走廊</option><option value="val_new_hallway_210">走廊后段</option></select>
<span id="buttons"></span><p id="label"></p><div class="row"><div>实际参与颜色合并的轮廓<img id="contour" alt="实际轮廓"></div><div><span id="colorLabel">本次颜色结果</span><img id="color" alt="颜色结果"></div></div>
<p><button onclick="side('previous','上一版')">上一版</button><button onclick="side('source','原片')">原片</button><button onclick="side('teacher','教师参考')">教师参考</button><button onclick="show()">本次结果</button></p>
<img src="comparison.png" alt="教师与修改前后同位置比较"><p><a href="student_RGB_fp32.pt">实际使用的FP32权重</a> · <a href="pipeline.json">本次完整处理配置</a> · <a href="stills_verification.json">静帧检查</a> · <a href="video_verification.json">视频检查</a> · <a href="REPORT.md">修改说明与局限</a></p>
<script>let grade=4;for(let i=1;i<=7;i++){let b=document.createElement('button');b.textContent=i+'档';b.onclick=()=>{grade=i;show()};buttons.appendChild(b)}function show(){let s=scene.value;contour.src=s+'_g'+grade+'_contour.png';color.src=s+'_g'+grade+'_color.png';label.textContent=grade+'档（1轻、7强）';colorLabel.textContent='本次颜色结果'}function side(k,t){color.src=scene.value+'_'+k+'.png';colorLabel.textContent=t}function changeMovie(){v.pause();v.src=movie.value+'_middle_12s_1080p.mp4';v.load()}show()</script></html>'''
    (out/'preview.html').write_text(html,encoding='utf-8')


@torch.no_grad()
def main():
    w=configure_workspace();contract_guard();set_reproducible(2026100487)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    tests=checks();out=RUN/'delivery';out.mkdir(exist_ok=False)
    old_model=ROOT/'native-stroke-color-texture-v4/generator_fp32.pt'
    state=torch.load(old_model,map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model'])
    shutil.copy2(old_model,out/'student_RGB_fp32.pt');shutil.copy2(OLD/'color_adapter.npz',out/'color_adapter.npz')
    shutil.copy2(OLD/'data.json',out/'data.json');data=json.loads((out/'data.json').read_text())
    adapter=np.load(out/'color_adapter.npz')['lab_chroma_affine']
    source_files=['coarse_surface_fill.py',Path(__file__).name,'contour_guided_fill.py','coarse_contour_targets.py',
                  'stroke_color_input_net.py','stroke_color_net.py','native_contour_net.py','train_context_contours.py',
                  'fpga_video_resize.py']
    code={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in source_files}
    config=dict(levels=LEVELS,color=COLOR,default='g4',precision='FP32, no AMP/TF32',
        sequence=['1080 integer area to 720','real learned student RGB prediction','uniform vector-TV removes fine texture',
                  'Canny/components from simplified surface','boundary-weighted vector-TV with simplified surface as fidelity target',
                  'frozen train-only Lab adapter + fixed cool/chroma/contrast adjustment','integer bilinear 720 to 1080'],
        source_weight=str(old_model),weight_sha256=file_hash(old_model),adapter_sha256=file_hash(out/'color_adapter.npz'),
        code=code,new_neural_training=False,weights_only=False,board_validated=False,
        no_teacher_at_inference=True,no_original_detail_bypass=True,no_temporal_blend=True,
        no_per_frame_histogram=True,no_synthetic_texture=True,
        parameter_selection='reviewed regression images; not a blind generalization result')
    write_json(out/'pipeline.json',config);write_json(out/'operator_checks.json',tests)
    records=[];crops=[]
    for row in data['validation']:
        key=row['id'];source=load_rgb(row['source']);pred=to_rgb(net(tensor(source)[None].cuda())[0])
        assert np.array_equal(pred,load_rgb(OLD/(key+'_before.png')))
        Image.fromarray(source).save(out/(key+'_source.png'));Image.fromarray(pred).save(out/(key+'_before.png'))
        shutil.copy2(row['target'],out/(key+'_teacher.png'));shutil.copy2(OLD/(key+'_g4_color.png'),out/(key+'_previous.png'))
        for level in LEVELS:
            rgb,e,stages=process(pred,adapter,level,True)
            Image.fromarray(rgb).save(out/(key+'_'+level+'_color.png'))
            Image.fromarray(np.uint8((1-e)*255)).save(out/(key+'_'+level+'_contour.png'))
            if level=='g4':
                for name,a in stages.items():Image.fromarray(a).save(out/(key+'_'+name+'.png'))
            records.append(dict(id=key,level=level,**edge_info(e),color=describe(rgb)))
        print('COARSE_STILLS',key,flush=True)
    for key,box,name in [('val_stairs_090',(200,420,440,660),'grass'),('val_stairs_090',(480,220,720,400),'building'),
                         ('val_plaza_090',(80,400,400,680),'people_trees'),('val_new_hallway_120',(0,0,320,280),'bright_window')]:
        x0,y0,x1,y1=box;rec=dict(id=key,box=box,name=name,outputs={})
        for s in ['source','teacher','previous','g4_color']:
            rec['outputs'][s]=describe(load_rgb(out/(key+'_'+s+'.png'))[y0:y1,x0:x1])
        for s,p in [('previous',OLD/(key+'_g4_contour.png')),('new',out/(key+'_g4_contour.png'))]:
            e=1-np.array(Image.open(p).convert('L'),np.float32)/255
            rec[s+'_contour']=edge_info(e[y0:y1,x0:x1])
        crops.append(rec)
    write_json(out/'stills_verification.json',dict(records=records,crops=crops,reloaded_RGB_all6_exact=True))
    contact(out)
    clips=json.loads((w/'runs/clear-style-control-20261003/video_clips.json').read_text())['clips'];reports=[]
    for clip_id in ['stairs','new_hallway']:
        clip=next(c for c in clips if c['id']==clip_id)
        assert file_hash(clip['clip'])==clip['clip_sha256'] and clip['source'] not in data['training_sources']
        cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        dest=out/(clip_id+'_middle_12s_1080p.mp4');pipe=encode(dest)
        previous=None;n=0;dups=0;motion=[];matched=[];start=time.perf_counter()
        validation={r['frame']:r for r in data['validation'] if r['clip']==clip['clip']}
        try:
            while n<360:
                ok,bgr=cap.read()
                if not ok:break
                source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                pred=to_rgb(net(tensor(source)[None].cuda())[0]);rgb,e=process(pred,adapter,'g4')
                pipe.stdin.write(bilinear_2to3(rgb).tobytes())
                if n in validation:
                    r=validation[n]
                    assert np.array_equal(rgb,load_rgb(out/(r['id']+'_g4_color.png')))
                    assert np.array_equal(source,load_rgb(out/(r['id']+'_source.png')))
                    matched.append(n)
                if previous is not None:
                    mx,my,valid=motion_map(previous[0],source);rec=dict(frame=n,valid_fraction=float(valid.mean()))
                    for label,a,b in [('original',previous[0],source),('raw_student',previous[1],pred),('new',previous[2],rgb)]:
                        warp=cv2.remap(a.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        err=np.abs(b.astype(np.float32)-warp).mean(2)
                        rec[label]=dict(mean=float(err[valid].mean()),p95=float(np.percentile(err[valid],95)),
                                        fraction_over12=float((err[valid]>12).mean()))
                    motion.append(rec);dups+=int(np.array_equal(previous[2],rgb))
                previous=(source,pred,rgb);n+=1
                if n%30==0:print('COARSE_VIDEO',clip_id,n,round(time.perf_counter()-start,1),flush=True)
        finally:
            cap.release();close_pipe(pipe)
        assert n==360 and len(matched)==2 and dups==0
        report=dict(id=clip_id,video=video_check(dest,n),source=clip,duplicates=dups,matched_native_frames=matched,motion=motion,
                    motion_mean={label:{m:float(np.mean([r[label][m] for r in motion])) for m in ['mean','p95','fraction_over12']} for label in ['original','raw_student','new']})
        reports.append(report);write_json(out/'video_verification.json',dict(clips=reports,temporal_acceptance=False,
            note='Source-flow residual is a diagnostic, not a proof that every region is flicker-free.'))
    assert all(file_hash(p)==h for p,h in code.items())
    (out/'source').mkdir()
    for name in source_files:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    page(out)
    print('COARSE_DELIVERY_DONE',out,flush=True)


if __name__=='__main__':main()
