"""A local, full-resolution comparison of actual StyleID teacher outputs."""
import argparse
import html
import json
from pathlib import Path
import shutil

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    run = args.root.resolve()
    sections = []
    titles = {"plaza_stone": "广场：地面石材与远景", "park_people": "公园：人物衣物、地面与树木", "quiet_stairs": "空旷台阶：石材、栏板与建筑"}
    for report in sorted(run.glob("*/report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        for row in sorted(data["records"], key=lambda r: {"quiet_stairs": 0, "plaza_stone": 1, "park_people": 2}.get(r["case"]["name"], 3)):
            name = row["case"]["name"]
            folder = report.parent / name
            rel = folder.relative_to(run).as_posix()
            stable = run.parent / "coherent-material-20261003/validation_review" / (Path(row["case"]["content"]["source"]).stem + "_0.png")
            if stable.is_file():
                shutil.copyfile(stable, folder / "previous_4850.png")
            choices = [("原始实拍", "source.png"), ("上一版4850", "previous_4850.png"), ("不加风格的反演重建", "reconstruction.png")]
            choices += [(f"StyleID γ={x['gamma']}", Path(x["file"]).name) for x in row["outputs"]]
            buttons = "".join(f'<button data-src="{rel}/{file}">{html.escape(label)}</button>' for label, file in choices if (folder / file).is_file())
            first = Path(row["outputs"][0]["file"]).name
            sections.append(f'''<section><h2>{titles.get(name, name)} · {html.escape(report.parent.name)}</h2>
<p>源图和输出均为 {row['size'][0]}×{row['size'][1]}。每次反演/生成实际 {len(data.get('timesteps',list(range(row['steps']))))} 步。下方可切换结果，并拖动分界线与原图对照。</p>
<div class="buttons">{buttons}</div><div class="labels"><b>原始实拍</b><b class="current">StyleID γ={row['outputs'][0]['gamma']}</b></div>
<div class="compare"><img class="before" src="{rel}/source.png"><img class="after" src="{rel}/{first}"><div class="divider"></div></div>
<input aria-label="原图与结果分界线" type="range" min="0" max="100" value="50">
<p><a class="full" href="{rel}/{first}" target="_blank">打开当前720p无损原图</a> · <a href="{rel}/comparison.jpg" target="_blank">四栏总览</a></p>
<details><summary>本次使用的真实游戏参考（已裁去周边UI）</summary><img class="reference" src="{rel}/reference.png"></details></section>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StyleID 材质试验 · 原图对照</title><style>
body{margin:0;background:#10151b;color:#e4ebf2;font:17px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1360px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:23px}p{color:#b8c6d4}.notice{padding:15px 20px;background:#253241;border-left:4px solid #e8bb63}section{margin:36px 0 50px;padding-top:12px;border-top:1px solid #394754}.buttons{display:flex;gap:10px;flex-wrap:wrap}button{background:#263340;color:white;border:1px solid #536477;padding:10px 16px;border-radius:6px;font:inherit;cursor:pointer}button.active{background:#8b6330}.labels{display:flex;justify-content:space-between;margin:12px 0}.compare{position:relative;aspect-ratio:16/9;background:black;overflow:hidden}.compare img{width:100%;height:100%;position:absolute;inset:0;object-fit:contain}.after{clip-path:inset(0 0 0 50%)}.divider{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid #fff;pointer-events:none}input{width:100%;margin:16px 0}a{color:#a8d4ff}.reference{max-width:100%;max-height:720px}summary{cursor:pointer}footer{color:#8897a5}
</style><main><h1>StyleID：真正的参考风格迁移试验</h1><div class="notice">这是预训练教师的单张图片试验，尚未通过材质与结构验收；不是新训练的共享CNN、连续视频或可上板权重。旧版4850保留。图片使用PNG，避免视频码率影响判断。</div>'''
    page += "\n".join(sections)
    page += '''<footer>作者算法：<a href="https://github.com/jiwoogit/StyleID">jiwoogit/StyleID</a> · 本地SD1.5反演和注意力注入；不同γ用于检查风格与内容保留的取舍。1080p放大图不代表新增原生细节。</footer></main><script>
document.querySelectorAll('section').forEach(s=>{const range=s.querySelector('input'),after=s.querySelector('.after'),line=s.querySelector('.divider');range.addEventListener('input',()=>{after.style.clipPath=`inset(0 0 0 ${range.value}%)`;line.style.left=range.value+'%'});s.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{after.src=b.dataset.src;s.querySelector('.full').href=b.dataset.src;s.querySelector('.current').textContent=b.textContent;s.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));}));});
</script></html>'''
    (run / "index.html").write_text(page, encoding="utf-8")
    for image in run.glob("*/**/*.png"):
        with Image.open(image) as im:
            im.verify()
    print(run / "index.html")


if __name__ == "__main__":
    main()
