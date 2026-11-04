"""Build a local new/previous/source comparison page with exact source binding."""
import argparse
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw
from .runtime import file_hash,write_json


NAMES={'train_escalator':'扶梯','holdout_corridor':'走廊','validation_plaza':'广场',
       'test_street':'街道','test_night':'夜间检查'}


def link_checked(src,dst,digest):
    if not dst.exists():os.link(src,dst)
    if file_hash(dst)!=digest:raise ValueError('Alias drift')


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--review-note',default='本轮评审版：暖暗氛围有所加强，树冠和人物边界仍有亮边，尚不能认定材质风格与视频稳定性已经达标。请结合完整运动画面对比。')
    a=p.parse_args()
    data=json.loads(a.render.read_text(encoding='utf-8'));old=json.loads(a.previous.read_text(encoding='utf-8'))
    out=a.render.parent;previous={r['id']:r for r in old['clips']};clips=[]
    board=Image.new('RGB',(1920,380*len(previous)),'#111');draw=ImageDraw.Draw(board)
    for index,row in enumerate(data['clips']):
        source=out/f"{row['id']}_source.mp4";link_checked(row['clip'],source,row['clip_sha256'])
        older=None
        if row['id'] in previous:
            earlier=previous[row['id']]
            if row['clip_sha256']!=earlier['clip_sha256']:raise ValueError('Comparison source mismatch')
            older=out/f"{row['id']}_previous.mp4";link_checked(earlier['output'],older,earlier['output_sha256'])
            current_panel=Image.open(out/f"{row['id']}_075_compare.jpg")
            older_image=Image.open(Path(earlier['output']).parent/f"{row['id']}_075_styled.png")
            images=[current_panel.crop((0,0,1920,1080)),older_image,current_panel.crop((1920,0,3840,1080))]
            for j,im in enumerate(images):
                board.paste(im.resize((640,360),Image.Resampling.LANCZOS),(640*j,380*index+20))
                draw.text((640*j+5,380*index+3),f"{row['id']} | {('SOURCE','PREVIOUS MODEL','NEW MODEL')[j]}",fill='white')
        clips.append({'name':NAMES.get(row['id'],row['id']),'source':source.name,
            'previous':older.name if older else None,'current':Path(row['output']).name})
    board.save(out/'previous_new_comparison.jpg',quality=95)
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>末日风格 · 新旧视频对比</title><style>
body{margin:0;background:#111418;color:#e8ecf1;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:0}p{color:#bec8d3}video{width:100%;background:#000;border-radius:10px}button{padding:10px 15px;margin:4px;border:1px solid #586474;border-radius:8px;color:#eee;background:#25303c;cursor:pointer}button.active{background:#8e6632}button:disabled{opacity:.4;cursor:default}section{margin-top:30px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}h2{font-size:19px}a{color:#b4d4ff}.note{font-size:14px}@media(max-width:800px){.pair{grid-template-columns:1fr}}
</style><main><h1>本轮末日风格效果</h1><p>完整视频 DURATION 秒，每个场景 5 秒。下方可同步比较上一版与这一版，也可以切回原片。</p>
<button id="fullplay">从头播放完整视频</button><button id="fullpause">暂停完整视频</button><video id="full" src="FULL_VIDEO" controls muted loop playsinline></video>
<section><h2>同一原片，新旧效果对比</h2><div id="tabs"></div><div><button id="oldmode">对照：上一版</button><button id="sourcemode">对照：原片</button></div>
<button id="play">同步播放</button><button id="pause">同时暂停</button><button id="restart">从头同步播放</button>
<div class="pair"><div><h2 id="referenceTitle">上一版</h2><video id="reference" controls muted playsinline></video></div><div><h2>这一版</h2><video id="current" controls muted playsinline></video></div></div>
<p class="note" id="note"></p><p class="note">同一套权重、固定参数。1080p 输入 → 720p 模型处理 → 1080p 输出。可全屏检查衣服、脸、墙面和运动中的细节。</p>
<p class="note">REVIEW_NOTE</p>
<p><a href="previous_new_comparison.jpg">原片／上一版／这一版静帧</a> · <a href="shared_style_fp32.pt" download>本轮真实 FP32 权重</a></p></section></main>
<script>const clips=CLIPS;let chosen=0,mode='previous';const ref=document.querySelector('#reference'),cur=document.querySelector('#current'),full=document.querySelector('#full');
function load(){ref.pause();cur.pause();const c=clips[chosen];const old=mode==='previous'&&c.previous;ref.src=old?c.previous:c.source;cur.src=c.current;document.querySelector('#referenceTitle').textContent=old?'上一版':'原片';document.querySelector('#oldmode').disabled=!c.previous;document.querySelector('#oldmode').classList.toggle('active',!!old);document.querySelector('#sourcemode').classList.toggle('active',!old);document.querySelectorAll('#tabs button').forEach((b,i)=>b.classList.toggle('active',i===chosen));document.querySelector('#note').textContent=c.previous?'这四段沿用上一版完全相同的源帧，用于检查改动。':'这段夜间素材未进入训练，用于额外检查；没有上一版视频对照。';}
clips.forEach((c,i)=>{let b=document.createElement('button');b.textContent=c.name;b.onclick=()=>{chosen=i;load();};document.querySelector('#tabs').append(b);});load();
document.querySelector('#oldmode').onclick=()=>{mode='previous';load();};document.querySelector('#sourcemode').onclick=()=>{mode='source';load();};
function play(){cur.currentTime=ref.currentTime;Promise.all([ref.play(),cur.play()]).catch(()=>{});}
document.querySelector('#play').onclick=play;document.querySelector('#pause').onclick=()=>{ref.pause();cur.pause();};document.querySelector('#restart').onclick=()=>{ref.currentTime=0;cur.currentTime=0;play();};
ref.ontimeupdate=()=>{if(!ref.paused&&Math.abs(ref.currentTime-cur.currentTime)>.12)cur.currentTime=ref.currentTime;};ref.onended=()=>{ref.pause();cur.pause();};document.querySelector('#fullplay').onclick=()=>{full.currentTime=0;full.play();};document.querySelector('#fullpause').onclick=()=>full.pause();</script></html>'''
    import html
    content=content.replace('这四段沿用上一版完全相同的源帧','这些片段沿用上一版完全相同的源帧')
    content=content.replace('DURATION',str(len(clips)*5)).replace('FULL_VIDEO',Path(data['combined_video']).name).replace('CLIPS',json.dumps(clips,ensure_ascii=False)).replace('REVIEW_NOTE',html.escape(a.review_note))
    (out/'index.html').write_text(content,encoding='utf-8')
    write_json(out/'comparison_binding.json',{'current_render_sha256':file_hash(a.render),
        'previous_render_sha256':file_hash(a.previous),'current_weights_sha256':data['weights_sha256'],
        'previous_weights_sha256':old['weights_sha256'],'clips':clips})
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
