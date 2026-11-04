"""Create a local, dependency-free video review page for trained model outputs."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
from .runtime import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.render.read_text(encoding='utf-8'));out=a.render.parent
    names={'train_escalator':'扶梯 · 训练视频','holdout_corridor':'走廊 · 保留测试',
           'validation_plaza':'广场 · 验证视频','test_street':'街道 · 测试视频'}
    clips=[]
    for row in data['clips']:
        target=out/f"{row['id']}_source_1080p.mp4"
        if not target.exists():os.link(row['clip'],target)
        if file_hash(target)!=row['clip_sha256']:raise ValueError('Source alias drift')
        clips.append({'name':names[row['id']],'source':target.name,'output':Path(row['output']).name})
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>共享模型视频效果</title><style>
body{margin:0;background:#101318;color:#edf0f5;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 8px}p{color:#b8c3d3}
video{width:100%;background:black;border-radius:10px}button{border:1px solid #405168;background:#1c2634;color:#e6efff;padding:10px 16px;border-radius:8px;margin:4px;cursor:pointer}
button.selected{background:#276bc0}section{margin-top:30px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}h2{font-size:19px;margin:6px 0}
a{color:#9fc7ff}@media(max-width:850px){.pair{grid-template-columns:1fr}}.note{font-size:14px}
</style><main><h1>同一模型，多场景效果</h1><p>完整效果共 20 秒，每个场景 5 秒。可点视频右下角全屏查看。</p>
<button id="fullplay">从头播放完整 20 秒</button><button id="fullpause">暂停完整视频</button>
<video id="full" controls muted loop playsinline src="all_scenes_1080p30_20s.mp4"></video>
<section><h2>原片与效果同步对比</h2><div id="tabs"></div>
<button id="play">同步播放</button><button id="pause">同时暂停</button><button id="restart">从头播放</button>
<div class="pair"><div><h2>原片</h2><video id="source" controls muted playsinline></video></div>
<div><h2>模型输出</h2><video id="styled" controls muted playsinline></video></div></div>
<p class="note">全部场景使用同一权重和固定参数。输入 1080p → 模型处理 720p → 输出 1080p。请同时看人物、材质纹理和连续运动。</p>
<p><a href="shared_style_fp32.pt" download>下载本次真实 FP32 权重</a> · <a href="all_scenes_comparison.jpg">查看静帧对比</a></p></section></main>
<script>const clips=CLIP_DATA;const a=document.querySelector('#source'),b=document.querySelector('#styled');
const full=document.querySelector('#full');document.querySelector('#fullplay').onclick=()=>{full.currentTime=0;full.play();};document.querySelector('#fullpause').onclick=()=>full.pause();
function choose(i){a.pause();b.pause();a.src=clips[i].source;b.src=clips[i].output;document.querySelectorAll('#tabs button').forEach((x,j)=>x.classList.toggle('selected',i===j));}
clips.forEach((c,i)=>{const x=document.createElement('button');x.textContent=c.name;x.onclick=()=>choose(i);document.querySelector('#tabs').append(x);});choose(0);
document.querySelector('#play').onclick=()=>{b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});};
document.querySelector('#pause').onclick=()=>{a.pause();b.pause();};document.querySelector('#restart').onclick=()=>{a.currentTime=0;b.currentTime=0;a.play();b.play();};
a.ontimeupdate=()=>{if(!a.paused&&Math.abs(a.currentTime-b.currentTime)>.12)b.currentTime=a.currentTime;};
a.onended=()=>{a.pause();b.pause();};</script></html>'''
    content=content.replace('CLIP_DATA',json.dumps(clips,ensure_ascii=False))
    (out/'index.html').write_text(content,encoding='utf-8')
    print(out/'index.html')


if __name__=='__main__':main()
