"""Build same-time full-frame media comparisons without copying old media."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .runtime import configure_workspace,write_json


def main():
    root=configure_workspace();runs=root/'runs';run=runs/'texture-recovery-20261003';old=runs/'clear-style-control-20261003'
    out=run/'review';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    cases=['quiet_stairs','plaza_stone','park_people'];names=['原图','上轮2000步学生','本轮预训练网络微调240步']
    records=[]
    for case in cases:
        paths=[old/'anime'/case/'source.png',old/'student_train/validation_002000'/f'{case}.png',run/'refine_train/validation_000240'/f'{case}.png']
        imgs=[Image.open(p).convert('RGB') for p in paths]
        board=Image.new('RGB',(1920,410),'#161b22');draw=ImageDraw.Draw(board)
        full=Image.new('RGB',(1280,3*770),'#161b22');fd=ImageDraw.Draw(full)
        for i,(name,im) in enumerate(zip(names,imgs)):
            draw.text((i*640+10,8),name,font=font,fill='white');board.paste(im.resize((640,360),Image.Resampling.LANCZOS),(i*640,50))
            fd.text((10,i*770+8),name,font=font,fill='white');full.paste(im,(0,i*770+50))
        board.save(out/(case+'_overview.png'));full.save(out/(case+'_native720.png'))
        records.append({'case':case,'source_paths':[str(p) for p in paths]})
    # Native pixel crops include the difficult regions, not only favorable surfaces.
    regions=[('quiet_stairs',(360,450,616,706),'台阶与植物'),('plaza_stone',(250,410,506,666),'人物与衣服'),('park_people',(0,350,256,606),'侧脸与植被')]
    detail=Image.new('RGB',(768,3*300),'#161b22');dd=ImageDraw.Draw(detail)
    for j,(case,box,title) in enumerate(regions):
        paths=[old/'anime'/case/'source.png',old/'student_train/validation_002000'/f'{case}.png',run/'refine_train/validation_000240'/f'{case}.png']
        for i,p in enumerate(paths):
            dd.text((i*256+6,j*300+6),['原图','上轮学生','本轮'][i]+' · '+title,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16),fill='white')
            detail.paste(Image.open(p).crop(box),(i*256,j*300+40))
    detail.save(out/'native_details.png')
    write_json(out/'image_sources.json',{'comparisons':records,'native_crop_boxes':regions})
    oldrel='../clear-style-control-20261003/'
    clips=[]
    for name,title,seconds in [('stairs','门廊、台阶与室内',12),('new_hallway','走廊与平滑墙面',12),('plaza','人物广场压力场景',15)]:
        clips.append({'id':name,'title':title,'seconds':seconds,'refined':f'refined_videos/{name}_refined_1080p_{seconds}s.mp4',
                      'student':oldrel+f'student_videos/{name}_student_1080p_{seconds}s.mp4',
                      'source':oldrel+f'video_sources/{name}_source_1080p.mp4',
                      'teacher':oldrel+f'teacher_videos/{name}_teacher_1080p_{seconds}s.mp4'})
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>纹理改进 · 同帧视频对照</title>
<style>body{margin:24px auto;max-width:1280px;padding:0 18px;background:#171b21;color:#f1f4f8;font:16px/1.6 system-ui}h1{font-size:25px}p{color:#cbd3df}button,select{padding:10px;margin:4px;background:#25374a;color:white;border:1px solid #597b9e;border-radius:7px}video,img{width:100%;background:black}video{max-height:72vh}a{color:#91caff}details{margin:20px 0}.row{display:flex;flex-wrap:wrap;gap:6px}#label{color:#b4d6ff}#error{color:#ffa898}</style>
<h1>纹理改进 · 同帧视频对照</h1>
<p>本轮保留预训练动画风格，再做240步720p完整帧对微调。上轮为效果偏弱的2000步学生。切换版本会保持同一播放位置；可暂停后反复切换或全屏查看。</p>
<div class="row"><label>场景 <select id="scene" aria-label="场景"></select></label><label>版本 <select id="version" aria-label="版本"><option value="refined">本轮：预训练网络微调240步</option><option value="student">上轮：2000步学生</option><option value="source">原视频</option><option value="teacher">作者预训练：未微调</option></select></label><button id="play">播放 / 暂停</button></div>
<div id="label"></div><div id="error" role="alert"></div><video controls loop playsinline preload="auto"></video>
<p>每帧实际计算，1080→720→1080；没有插帧、原片混合或贴回人脸。此轮是明显动画风格验证，小脸及密集植被仍需审查，不能视为消光写实或上板成果。30fps是播放帧率。</p>
<p><a href="final/refined_style_fp32.pt" download>真实FP32权重</a> · <a href="final/README.md">复现说明</a> · <a href="RESULTS.md">结论与限制</a></p>
<details open><summary>三版本整图：原图 / 上轮学生 / 本轮</summary><img src="review/quiet_stairs_overview.png"><p><a href="review/quiet_stairs_native720.png">打开完整720p对照</a></p></details>
<details><summary>人物与植被整图</summary><img src="review/plaza_stone_overview.png"><img src="review/park_people_overview.png"><p><a href="review/plaza_stone_native720.png">广场完整720p</a> · <a href="review/park_people_native720.png">公园完整720p</a></p></details>
<details><summary>原生像素局部：同时看纹理和失败区域</summary><img style="max-width:768px" src="review/native_details.png"></details>
<script>const clips=__CLIPS__;
const scene=document.querySelector('#scene'),version=document.querySelector('#version'),v=document.querySelector('video');
clips.forEach((r,i)=>scene.add(new Option(r.title+' · '+r.seconds+'秒',i)));
function load(keep=false){const t=keep?v.currentTime:0,playing=!v.paused,r=clips[+scene.value];document.querySelector('#error').textContent='';v.src=r[version.value];document.querySelector('#label').textContent=r.title+' / '+version.options[version.selectedIndex].text+' / '+r.seconds+'秒';v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration-.1);if(playing)v.play().catch(e=>document.querySelector('#error').textContent=e.message)};}
scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();v.onerror=()=>document.querySelector('#error').textContent='视频加载失败：'+v.currentSrc;load();</script></html>'''
    (run/'index.html').write_text(html.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False)),encoding='utf-8')
    print('GALLERY_BUILT',flush=True)


if __name__=='__main__':main()
