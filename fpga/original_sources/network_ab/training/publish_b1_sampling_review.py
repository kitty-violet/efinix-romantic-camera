"""Present actual integer outputs next to the preserved fallback, without edits."""
import argparse
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .audit_b1_sampling_v2 import FALLBACK,OUT
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--videos',type=Path,required=True);a=ap.parse_args()
    v2=json.loads((a.videos/'results.json').read_text(encoding='utf-8'))
    assert {r['id'] for r in v2['results']}=={'stairs','plaza'}, 'Both complete videos are required before publishing'
    board=Image.open(a.package/'v1_v2_teacher.png').convert('RGB')
    # Both columns are unscaled 720p network crops. Headers are replaced only.
    short=Image.new('RGB',(840,624),'#17202a');draw=ImageDraw.Draw(short);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for row,source_row in enumerate((2,3)):
        for col,source_col in enumerate((1,2)):
            short.paste(board.crop((source_col*420,source_row*312+32,(source_col+1)*420,source_row*312+312)),(col*420,row*312+32))
            draw.text((col*420+6,row*312+5),('V1兜底','本次实际小网络')[col],font=font,fill='white')
    short.save(a.package/'trees_people_v1_v2.png')
    v1=json.loads((FALLBACK/'videos/results.json').read_text(encoding='utf-8'))
    checks=[];sections=[]
    rel=lambda p:Path(os.path.relpath(p,a.videos)).as_posix()
    for r in v2['results']:
        old=next(q for q in v1['results'] if q['id']==r['id']);key=r['id']
        assert old['source_sha256']==r['source_sha256'] and old['video']['frames']==r['video']['frames']
        stats={}
        for label,record in [('v1',old),('v2',r)]:
            rows=record['motion'];stats[label]={f:sum(m['output'][f] for m in rows)/len(rows) for f in ('mean','p95','over16')}
        checks.append(dict(scene=key,matched_source_sha256=r['source_sha256'],motion=stats,encoding=r['encoding']))
        items=[]
        for label,p in [('原片',Path(r['source'])),('V1兜底',FALLBACK/'videos'/f'{key}_integer1080.mp4'),('新版实际小网络',a.videos/f'{key}_integer1080.mp4')]:
            items.append(f'<section><h3>{label}</h3><video controls muted preload="metadata" src="{rel(p)}"></video></section>')
        sections.append(f'<article><h2>{"台阶、草丛" if key=="stairs" else "人物、树冠"} · {r["video"]["seconds"]:g} 秒</h2><button onclick="playGroup(this)">同步播放 / 暂停</button><div class="grid">'+''.join(items)+'</div></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B1 新版与第一版兜底</title><style>body{background:#101923;color:#edf1f5;margin:24px;font:16px system-ui;line-height:1.6}h1{font-size:25px}h2{font-size:22px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video,img{width:100%}button,a{color:#91d5ff}button{background:#234155;border:1px solid #789;border-radius:5px;padding:7px 15px;cursor:pointer}section{min-width:0}article{margin:30px 0}.note{max-width:1000px}figure{margin:0;max-width:1260px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><h1>新版实际小网络 / V1兜底</h1><p class="note">原片和新版按同一1080→720→1080链路比较，视频为12秒和15秒。新版是实际20 KiB权重包的软件整数运行结果，尚未上板；没有追加图像滤镜。V1保留作为备用。请全屏检查人物、树冠和建筑边缘，并比较运动时是否出现新的闪动。</p>'''+''.join(sections)+f'<h2>原尺寸局部对照</h2><figure><img src="{rel(a.package/"v1_v2_teacher.png")}"></figure>'+'''<script>function playGroup(b){const vs=[...b.closest('article').querySelectorAll('video')];const running=vs.some(v=>!v.paused);if(running){vs.forEach(v=>v.pause());return}const t=vs[2].currentTime;vs.forEach(v=>{v.currentTime=t;v.play().catch(()=>{});});}document.querySelectorAll('article').forEach(a=>{const vs=[...a.querySelectorAll('video')];vs[2].addEventListener('timeupdate',()=>{if(!vs[2].paused)vs.slice(0,2).forEach(v=>{if(Math.abs(v.currentTime-vs[2].currentTime)>.18)v.currentTime=vs[2].currentTime;});});});</script></html>'''
    (a.videos/'compare_v1_v2.html').write_text(page,encoding='utf-8')
    write_json(a.videos/'comparison.json',dict(package_sha256=file_hash(a.package/'coefficients-padded.bin'),
        records=checks,visual_accepted=False,actual_board=False,metrics_are_diagnostics_not_acceptance=True))
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
