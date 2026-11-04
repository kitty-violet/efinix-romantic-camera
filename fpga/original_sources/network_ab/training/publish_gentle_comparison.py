"""Publish and verify the small static correction, without rerendering a video."""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image
from .block_video_runtime import STILLS
from .fpga_video_resize import bilinear_2to3
from .gentle_block_boundaries import process
from .runtime import file_hash, write_json


def main():
    out = STILLS.parent / 'gentle-global-boundaries-20261005-v2'
    experiment = json.loads((out / 'experiment.json').read_text(encoding='utf-8'))
    records = []
    for row in experiment['records']:
        name = row['id']
        assert file_hash(row['input']) == row['input_sha256']
        before = np.array(Image.open(out / (name + '_before720.png')))
        after = np.array(Image.open(out / (name + '_after720.png')))
        assert before.shape == after.shape == (720, 1280, 3)
        assert np.array_equal(process(before), after)
        displayed = np.array(Image.open(out / (name + '_1080p.png')))
        assert np.array_equal(bilinear_2to3(after), displayed)
        Image.fromarray(bilinear_2to3(before)).save(out / (name + '_before1080p.png'))
        records.append(dict(id=name, exact_reproduction=True,
                            output_shape=list(displayed.shape),
                            input_sha256=row['input_sha256']))
    sources = out / 'source_code'
    sources.mkdir(exist_ok=True)
    for name in ['gentle_block_boundaries.py', 'publish_gentle_comparison.py',
                 'painting_parameters.py', 'block_separation.py', 'fpga_video_resize.py']:
        shutil.copy2(Path(__file__).with_name(name), sources / name)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>回到之前，只小幅收紧过渡</title>
<style>body{margin:0;background:#15191f;color:#eef1f4;font:16px system-ui}main{max-width:1500px;margin:auto;padding:20px}
h1{font-size:22px}p{color:#bfc8d2}button,select{font:inherit;padding:10px 16px;border:1px solid #657483;border-radius:6px;background:#26323e;color:white;cursor:pointer;margin:4px}
button.on{background:#356f94}img{display:block;width:100%;height:auto}#label{padding:10px 4px;font-weight:600}.bar{display:flex;align-items:center;flex-wrap:wrap}</style>
<main><h1>回到之前，只小幅收紧过渡</h1><p>点击切换同一帧。保留原来的颜色层次，轻微加强已有色块的分界。</p>
<div class="bar"><select id="scene" aria-label="场景"><option value="plaza_090">远处树木</option><option value="plaza_330">近处人物</option><option value="stairs_090">台阶与草丛</option><option value="hallway_120">走廊</option></select>
<button id="before">之前的画面</button><button id="after" class="on">本次小幅调整</button></div><div id="label">本次小幅调整</div><img id="picture" alt="本次小幅调整" src="plaza_090_1080p.png">
<p>本页为四张同帧静态对照；尚未验证本次调整的整段视频效果。</p></main>
<script>const scene=document.getElementById('scene'), picture=document.getElementById('picture'), label=document.getElementById('label');let mode='after';
function show(){picture.src=scene.value+(mode==='before'?'_before1080p.png':'_1080p.png');label.textContent=mode==='before'?'之前的画面':'本次小幅调整';picture.alt=label.textContent;for(const m of ['before','after'])document.getElementById(m).classList.toggle('on',mode===m)}
for(const m of ['before','after'])document.getElementById(m).onclick=()=>{mode=m;show()};scene.onchange=show;
for(const s of ['plaza_090','plaza_330','stairs_090','hallway_120'])for(const suffix of ['_before1080p.png','_1080p.png']){const i=new Image();i.src=s+suffix}
</script></html>'''
    (out / 'preview.html').write_text(html, encoding='utf-8')
    (out / 'REPORT.md').write_text('''# 小幅收紧过渡：静态对照

用户否定全局强量化，要求恢复之前画面，仅稍微减少颜色过渡并强调色块分界。

本次恢复未经过强量化的720p画面。广场两帧采用完整原生链路预检保存的before720（旧背景处理+原人物处理），台阶和走廊采用此前balanced原生输出。不是从压坏的图片修复回来，也不是原始实拍图。

取消新增TV平滑和大间距硬调色板。保留细色阶间距L=6、ab=4，以连续曲线少量收紧过渡，混合强度为0.40/0.24；仅半强度单次边界收紧，并限制每像素Lab总改变量为1.8/1/1。所有物体使用同一套固定参数。输入真实720p，在720p进行本次处理后用项目整数双线性上采样至1080p。

四个场景实际运行成功；记录输入SHA、参数、代码副本；独立重新计算逐像素一致，1080p输出与整数上采样逐像素一致。95%的RGB通道变化不超过2/255，平均变化0.51–0.81/255。这仅说明改动幅度小，不作为风格质量或动态稳定性判定。

已人工查看树木、人物、台阶和走廊同帧图片。这是软件静态外观调整，没有新训练或新权重，没有本次整段视频或时序稳定性验收，也没有融合FPGA。强全局量化两版仅做预检，已经被用户否定，未生成完整视频。
''', encoding='utf-8')
    write_json(out / 'verification.json', dict(records=records, video_rendered=False,
                                               temporal_validated=False, new_weights=False))
    manifest = [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size,
                     sha256=file_hash(p)) for p in sorted(out.rglob('*'))
                if p.is_file() and p.name != 'manifest.json']
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(dict(output=str(out), files=len(manifest),
                          bytes=sum(p['bytes'] for p in manifest), records=records), ensure_ascii=False))


if __name__ == '__main__':
    main()
