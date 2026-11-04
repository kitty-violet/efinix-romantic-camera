"""Local video/still review, with source/author/student kept explicitly separate."""
import json
from pathlib import Path
from .runtime import configure_workspace


def main():
    root=configure_workspace();run=root/'runs/clear-style-control-20261003'
    clips=json.loads((run/'video_clips.json').read_text(encoding='utf-8'))['clips'];rows=[]
    titles={'new_hallway':'简单走廊 · 隔离验证源','stairs':'门廊、台阶与室内 · 回归场景','plaza':'人物广场 · 压力场景'}
    for c in clips:
        row={'id':c['id'],'title':titles[c['id']],'seconds':c['validation']['seconds'], 'source':str(Path(c['clip']).relative_to(run)).replace('\\','/')}
        for kind in ['teacher','guarded','student']:
            report=run/(kind+'_videos/report.json')
            if report.exists():
                matches=[x for x in json.loads(report.read_text(encoding='utf-8'))['records'] if x['id']==c['id']]
                if matches:row[kind]=str(Path(matches[0]['output']).relative_to(run)).replace('\\','/')
        rows.append(row)
    student=all('student' in r for r in rows)
    guarded=all('guarded' in r for r in rows)
    options=('<option value="student">本轮2000步CNN</option>' if student else '')+('<option value="guarded">修正后的动画教师</option>' if guarded else '')+'<option value="teacher">作者原始模型</option><option value="source">原视频</option>'
    deliverables=''
    if (run/'final-2000/verification.json').exists():
        deliverables='<p><a href="final-2000/shared_style_fp32.pt" download>下载本轮真实FP32权重</a> · <a href="final-2000/README.md">独立推理说明</a> · <a href="RESULTS.md">实验结论与限制</a></p>'
    for case,label in [('quiet_stairs','台阶与植被'),('plaza_stone','广场人物'),('park_people','近景人物与草木')]:
        if (run/'review'/(case+'_overview.jpg')).exists():
            deliverables+=f'<details><summary>原图 / 教师 / 2000步CNN：{label}</summary><p><a href="review/{case}_native_comparison.png">原生720p整图对照</a> · <a href="review/{case}_native_detail.png">原生局部对照</a></p><img style="max-width:640px" src="review/{case}_overview.jpg"></details>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>明显风格对照 · 连续视频</title>
<style>body{margin:24px auto;max-width:1200px;padding:0 18px;font:16px/1.6 system-ui;background:#171b21;color:#f2f3f5}h1{font-size:25px}p{color:#ccd3de}select,button{padding:10px;margin:4px;background:#26374a;color:white;border:1px solid #597594;border-radius:6px}video{width:100%;max-height:72vh;background:black}img{width:100%}a{color:#91caff}details{margin:18px 0}#label{color:#c5ddff}.row{display:flex;gap:8px;flex-wrap:wrap}</style>
<h1>明显风格对照 · 连续视频</h1><p>每段12～15秒。同一原片可切换对比；点击视频右下角可全屏。处理链为1080→720→1080。</p>
<div class="row"><label>场景 <select id="scene" aria-label="场景"></select></label><label>版本 <select id="version" aria-label="版本">OPTIONS</select></label><button id="play">播放 / 暂停</button></div>
<div id="label"></div><video id="video" controls loop playsinline preload="auto"></video>
<p>作者模型为公开预训练结果；本轮CNN出现后单独标注。动画效果仅是迁移能力对照，不代表消光写实目标完成。人物细节与植被稳定性需结合完整视频判断。</p>
DELIVERABLES
<details><summary>静帧：同方法换参考与作者模型</summary><p>WCT²换动画参考仍偏调色；作者动画网络才出现明显轮廓和纹理简化。</p><img src="wct/quiet_stairs_overview.jpg"><img src="anime/quiet_stairs_overview.jpg"><img src="anime/plaza_stone_overview.jpg"><img src="anime/park_people_overview.jpg"></details>
<script>const clips=DATA;const scene=document.querySelector('#scene'),version=document.querySelector('#version'),v=document.querySelector('video');clips.forEach((r,i)=>scene.add(new Option(r.title+' · '+r.seconds+'秒',i)));let changing=false;
function load(keep=false){const t=keep?v.currentTime:0,playing=!v.paused;const r=clips[+scene.value],k=version.value;v.src=r[k];document.querySelector('#label').textContent=r.title+' / '+version.options[version.selectedIndex].text+' / '+r.seconds+'秒';v.onloadedmetadata=()=>{v.currentTime=Math.min(t,Math.max(0,v.duration-.1));if(playing)v.play()};}
scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();load();</script></html>'''
    (run/'index.html').write_text(page.replace('OPTIONS',options).replace('DELIVERABLES',deliverables).replace('DATA',json.dumps(rows,ensure_ascii=False)),encoding='utf-8')
    print('REVIEW_PAGE',run/'index.html','student_available',student)


if __name__=='__main__':main()
