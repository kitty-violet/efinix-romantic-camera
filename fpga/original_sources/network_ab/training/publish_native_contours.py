"""Deliver genuine contour-only weights, native maps and a 12s diagnostic video."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .native_contour_net import NativeContourNet
from .train_native_contours import ROOT, tensor, load_rgb, scores, edges
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .fpga_video_resize import area_3to2,bilinear_2to3
from .publish_region_classes import encode,motion_map
from .render_wavelet_game import close_pipe,video_check


def picture(out,run,data,step):
    ev=run/f'eval-{step:06d}'
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    specs=[('val_stairs_090',(420,180,740,460)),('val_stairs_090',(140,380,460,660)),('val_plaza_090',(80,400,400,680))]
    board=Image.new('RGB',(1280,3*314),'#19212b');draw=ImageDraw.Draw(board)
    for j,(name,box) in enumerate(specs):
        r=next(r for r in data['validation'] if r['id']==name)
        items=[('教师画面',r['target']),('教师边界标签',ev/(name+'_target.png')),
               ('上一版轮廓预测',ev/(name+'_previous.png')),('本次真实预测（未细化）',ev/(name+'_probability.png'))]
        for k,(title,path) in enumerate(items):
            draw.text((320*k+4,314*j+5),title,font=font,fill='white')
            board.paste(Image.open(path).convert('RGB').crop(box),(320*k,314*j+32))
    board.save(out/'comparison.png')


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100483);cv2.setNumThreads(2);torch.set_num_threads(4)
    run=a.run;state=torch.load(run/'contour_fp32.pt',map_location='cpu',weights_only=True)
    is_stroke=state['config'].get('label',{}).get('kind')=='teacher_dark_ridge'
    assert state['folded'] and all(file_hash(p)==h for p,h in state['binding']['code'].items())
    cls=NativeContourNet
    if state['config'].get('graph')=='context90_native_contour':
        from .train_context_contours import ContextContourNet
        cls=ContextContourNet
    net=cls(difference=False).cuda().eval();net.load_state_dict(state['model'])
    ck=torch.load(run/f'checkpoint_{state["step"]:06d}.pt',map_location='cpu',weights_only=True)
    ref=cls(difference=True).cuda().eval();ref.load_state_dict(ck['model'])
    data=json.loads((run/'data.json').read_text(encoding='utf-8'));out=run/'delivery';out.mkdir(exist_ok=False)
    checks=[]
    for r in data['validation']:
        x=tensor(load_rgb(r['source']))[None].cuda();z=net(x);diff=float((z-ref(x)).abs().max())
        assert diff<=1e-6
        p=z.sigmoid()[0,0].cpu().numpy();actual=np.uint8((1-p)*255)
        saved=np.array(Image.open(run/f'eval-{state["step"]:06d}'/(r['id']+'_probability.png')))
        pe=int(np.abs(saved.astype(int)-actual.astype(int)).max());assert pe==0
        checks.append(dict(id=r['id'],logit_max_error=diff,png_max_error=pe))
    del ref
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in rows]==list(range(1,state['step']+1))
    write_json(out/'weight_verification.json',dict(status='PASS',weights=str(run/'contour_fp32.pt'),sha256=file_hash(run/'contour_fp32.pt'),
        checks=checks,contiguous_updates=state['step'],native_updates=sum(r['native'] for r in rows),
        inference_class=cls.__name__+'(difference=False)',output='NCHW signed contour logit, no RGB style image',
        input='NCHW FP32 (RGB uint8-128)/128 at1280x720',fpga_validated=False,style_complete=False))
    picture(out,run,data,state['step'])
    root=configure_workspace();clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    clip=next(c for c in clips if c['id']=='stairs')
    assert clip['source'] not in data['training_sources'] and file_hash(clip['clip'])==clip['clip_sha256']
    cap=cv2.VideoCapture(clip['clip']);path=out/'contours_12s_1080p.mp4';pipe=encode(path)
    previous=None;motion=[];n=0;duplicates=0;start=time.perf_counter()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            source=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            p=net(tensor(source)[None].cuda()).sigmoid()[0,0].cpu().numpy()
            mono=np.uint8(np.clip(1-p,0,1)*255);rgb=np.repeat(mono[:,:,None],3,2)
            pipe.stdin.write(bilinear_2to3(rgb).tobytes())
            if n in (90,210):Image.fromarray(rgb).save(out/f'video_{n:03d}_native720.png')
            if previous is not None:
                mx,my,valid=motion_map(previous[0],source)
                warped=cv2.remap(previous[1],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                err=np.abs(p-warped);band=(p>.15)|(warped>.15);mask=valid&band
                motion.append(dict(frame=n,valid_fraction=float(valid.mean()),edge_band_fraction=float(mask.mean()),
                    probability_mae=float(err[valid].mean()),edge_band_probability_mae=float(err[mask].mean()) if mask.any() else None))
                duplicates+=int(np.array_equal(mono,previous[2]))
            previous=(source,p,mono);n+=1
            if n%60==0:print('CONTOUR_VIDEO',n,round(time.perf_counter()-start,1),flush=True)
    finally:
        cap.release();close_pipe(pipe)
    assert n==clip['validation']['frames'];check=video_check(path,n)
    write_json(out/'video.json',dict(check=check,frames=n,duration=n/30,duplicates=duplicates,source=clip,
        pipeline='1080 integer area ->720 contour network -> probability grayscale display ->integer bilinear1080',
        no_nms=True,no_temporal_blend=True,style_output=False,temporal_acceptance=False,
        mean_probability_warp_error=float(np.mean([r['probability_mae'] for r in motion])),
        mean_edge_band_probability_warp_error=float(np.mean([r['edge_band_probability_mae'] for r in motion if r['edge_band_probability_mae'] is not None])),
        motion_records=motion))
    teacher_records=json.loads((root/'runs/b1-distillation-20261003/videos-2000/report.json').read_text(encoding='utf-8'))['records']
    teacher=next(r for r in teacher_records if r['id']=='stairs')
    assert teacher['clip_sha256']==clip['clip_sha256'] and file_hash(teacher['teacher'])==teacher['teacher_validation']['sha256']
    source_url='/'+Path(clip['clip']).relative_to(root/'runs').as_posix()
    teacher_url='/'+Path(teacher['teacher']).relative_to(root/'runs').as_posix()
    metrics=json.loads((run/f'eval-{state["step"]:06d}'/'report.json').read_text(encoding='utf-8'))
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["source_canny"]["f1_exact"]:.3f}</td><td>{r["previous_model"]["f1_exact"]:.3f}</td><td>{r["predicted"]["f1_exact"]:.3f}</td><td>{r["predicted"]["precision_exact"]:.3f}</td><td>{r["predicted"]["recall_exact"]:.3f}</td></tr>' for r in metrics['records'] if not r['training_patch'])
    label_description=('标签提取教师画面中细小深色描线的强度，避免将一道线的两侧当成两条线。' if is_stroke else '标签是在教师画面上提取的Canny边界。')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>轮廓专项：真实预测</title>
<style>body{{background:#141b25;color:#e8eef5;font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 16px}}h1{{font-size:26px}}p{{line-height:1.7}}button,a{{color:#9fe3f1}}button{{background:#253344;border:1px solid #64788a;padding:10px;margin:4px;border-radius:5px}}video{{width:min(100%,1280px);display:block;background:#000}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 14px;border:1px solid #4c5c6b}}.warn{{color:#ffd38b}}a{{display:inline-block;margin:8px}}</style>
<h1>先把轮廓学清楚：{state['step']}步，{sum(p.numel() for p in net.parameters()):,}参数</h1>
<p>这页只检查轮廓。模型输入只有原片；预测没有Canny、细化、锐化或混帧。教师边界图由已认可的教师画面离线提取，只作为监督标签。{label_description}下方场景来自未参与训练的视频源，但此前已被多次查看，属于回归检查。</p>
<p class="warn">还不是完成的动画风格画面，也没有通过FPGA与视频稳定性验收。细线改善和草丛闭合分组需分别判断，不能只看数值。</p>
<button onclick="swap('contours_12s_1080p.mp4','本次真实轮廓')">本次轮廓</button><button onclick="swap('{source_url}','原片')">原片</button><button onclick="swap('{teacher_url}','教师画面')">教师画面</button><button onclick="v.currentTime=0;v.play()">从头播放12秒</button><button onclick="v.pause();v.currentTime=3">看3秒处</button><span id="label">本次真实轮廓</span>
<video id="v" src="contours_12s_1080p.mp4" controls preload="metadata"></video>
<p>真实连续360帧，30fps，1920×1080；1080→720网络→1080。视频中的灰度是预测的轮廓响应，不是RGB风格结果。</p>
<img src="comparison.png" alt="教师、边界标签、上一版和本次真实轮廓的原像素裁剪对比">
<p>逐像素边界评分：不允许位置偏差、不进行细化，固定阈值0.5。标签是教师图像提线，并非人工语义标注；不同标签的实验分数不能直接横比。原片直接提线也是必要对照。</p><table><tr><th>场景</th><th>原片提线F1</th><th>上一版F1</th><th>本次F1</th><th>精确率</th><th>召回率</th></tr>{table}</table>
<a href="../contour_fp32.pt">真实FP32轮廓权重</a><a href="weight_verification.json">重载验证</a><a href="../eval-{state['step']:06d}/comparison.png">完整对照图</a><a href="video.json">视频检查</a>
<p>方法来源：<a href="https://github.com/hellozhuo/pidinet">PiDiNet</a><a href="https://github.com/WHUHLX/CATS/blob/master/loss/tracingloss.py">CATS边界竞争约束</a>；借鉴局部方法，没有复现完整论文网络。</p>
<script>function swap(src,title){{const t=v.currentTime,playing=!v.paused;v.src=src;label.textContent=title;v.onloadedmetadata=()=>{{v.currentTime=Math.min(t,v.duration);if(playing)v.play()}};v.load()}}</script></html>'''
    (out/'preview.html').write_text(html,encoding='utf-8');print('CONTOUR_DELIVERY_DONE',flush=True)


if __name__=='__main__':main()
