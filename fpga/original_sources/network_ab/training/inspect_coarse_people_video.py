"""Decode and package the person-coarsening preview without rerendering it."""
import json,math,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .publish_coarse_people import OUT,ROOT,OLD,SELECTED
from .runtime import file_hash,write_json
from .block_video_runtime import STILLS


def main():
    verification=json.loads((OUT/'video_verification.json').read_text(encoding='utf-8'))
    config=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'))
    assert verification['complete'] and verification['frames']==450
    assert all(file_hash(p)==h for p,h in config['live_code_hashes'].items())
    # The colour helper is identical to the earlier frozen, approved still
    # recipe. Store its binding and adapter too, rather than relying on cwd.
    helper=Path(__file__).with_name('painting_parameters.py')
    frozen_helper=STILLS/'source_code/painting_parameters.py'
    assert file_hash(helper)==file_hash(frozen_helper)
    provenance={str(helper):file_hash(helper),str(STILLS/'color_adapter.npz'):file_hash(STILLS/'color_adapter.npz'),str(Path(__file__)):file_hash(Path(__file__))}
    for path in provenance:
        shutil.copy2(path,OUT/'source_code'/('extra_'+Path(path).name))
    write_json(OUT/'supporting_artifacts.json',dict(hashes=provenance,approved_still_recipe=str(STILLS),colour_helper_matches_frozen=True))
    path=OUT/'plaza_coarse_people_15s_1080p.mp4'
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
    n=0;rows=[];previous=None;duplicates=0
    while True:
        ok,bgr=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert rgb.shape==(1080,1920,3)
        if previous is not None:duplicates+=int(np.array_equal(previous,rgb))
        previous=rgb
        if n in SELECTED:
            reference=np.array(Image.open(OUT/f'frame_{n:03d}_1080p.png'))
            error=rgb.astype(np.float32)-reference.astype(np.float32)
            mse=float(np.square(error).mean());mae=float(np.abs(error).mean())
            rows.append(dict(frame=n,codec_mae_255=mae,psnr_db=10*math.log10(255**2/mse)))
            Image.fromarray(rgb).save(OUT/f'decoded_{n:03d}_1080p.png')
        n+=1
    cap.release();assert n==450 and duplicates==0
    assert all(r['codec_mae_255']<3 for r in rows)
    # Crop actual 1080p render pixels; no enlargement, artificial sharpening or
    # replacement teacher face is used in this side-by-side.
    cases=[(210,(195,645,660,1045)),(330,(965,420,1430,820)),(330,(0,660,465,1060))]
    board=Image.new('RGB',(930,len(cases)*436),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    for row,(frame,box) in enumerate(cases):
        for col,(prefix,title) in enumerate([('old','上一版：人物内部碎边较多'),('frame','本次：简化衣物，减轻脸部刻画')]):
            draw.text((col*465+8,row*436+4),title,font=font,fill='white')
            board.paste(Image.open(OUT/f'{prefix}_{frame:03d}_1080p.png').crop(box),(col*465,row*436+36))
    board.save(OUT/'people_comparison.png')
    write_json(OUT/'decode_verification.json',dict(frames=n,identical_adjacent_frames=duplicates,selected_codec_checks=rows,video_sha256=file_hash(path),bytes=path.stat().st_size))
    motion=verification['person_flow_diagnostic']
    report=f'''# 人物粗化预览

用户要求人物更粗糙、减少细碎刻画，同时不要扭曲。当前交付为15秒、450帧、30fps、1920×1080的对照预览，无音频。

## 本次改变

人物直接从真实学生RGB预测分支进行颜色面简化，绕开旧版人物的方向滤波、局部反差强化和粗黑描边。衣物TV强度0.10，肤色连续提示降低局部强度至最小0.014，保留主要五官；肤色提示只是连续色彩启发式，不能当作可靠人脸语义解析。固定色调和连续色级仍应用于人物，不还原原片RGB，不进行人物坐标变形。

LRASPP仅使用class15定位人物。光流仅稳定人物概率，不混合或扭曲输出RGB。遮挡、远小人和特殊肤色/光照仍需更多场景检验。

## 处理来源与边界

人物从原始SDR1080图像经整数area缩至720、真实FP32学生推理、人物色块处理、整数bilinear放至1080。背景复用上一版1080p已编码成片，再合成编码，所以本片存在背景二次压缩，并非全链重新渲染。alpha为0的区域在编码前与上一版解码图逐像素一致。

本轮未重新训练、未修改学生权重、没有教师推理。属于软件效果预览；附加分割及算子尚未融入FPGA网络，也没有资源、定点或时序验收。

## 核验

- 完整独立解码450帧，1080p/30fps，相邻重复帧0。
- 5个编码前后同帧检查MAE范围{min(r['codec_mae_255'] for r in rows):.3f}–{max(r['codec_mae_255'] for r in rows):.3f}/255，仅说明编码误差，不能据此声称清晰度或风格达标。
- 源光流对齐的人物区域差异均值：上一版{motion['previous']['mean']:.4f}，本次{motion['current']['mean']:.4f}/255。风格变化会影响这一指标，不能等同于无闪烁或无变形验收。
- CPU/GPU掩膜4静帧8bit差异0；30帧短检查320/330/349成片加速前后逐像素一致。
- source_code及pipeline.json保留源码快照和权重/输入绑定；旧视频和静帧未覆盖。

观看重点：7秒的人群、11秒靠近镜头的两个人及左侧花衣服人物。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    source='/' + Path(config['source_clip']['clip']).relative_to(ROOT.parent).as_posix()
    old='/' + OLD.relative_to(ROOT.parent).as_posix()
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人物粗化 · 15秒对照</title><style>body{background:#17202a;color:#e8f0f4;font:17px system-ui;margin:24px auto;padding:0 18px;max-width:1500px}h1{font-size:27px}p{line-height:1.65}button{padding:11px 15px;margin:4px;background:#28465c;color:white;border:1px solid #92acbb;border-radius:6px;cursor:pointer}video{width:100%;max-height:78vh;background:#000}img{max-width:100%}a{color:#a6e4ff}.note{color:#bdd0da}.row{margin:16px 0}#label{color:#ffdc98}</style>
<h1>人物减少细碎刻画，衣服归并成更大的色块</h1>
<p>同一段15秒视频，可在同一时间点切换。重点看7秒的人群、11秒近景人物的衣服和脸。</p>
<div class="row"><button onclick="swap('current')">本次人物粗化</button><button onclick="swap('previous')">上一版对照</button><button onclick="swap('source')">原视频</button><button onclick="v.currentTime=0;v.play()">从头播放15秒</button><button onclick="v.pause();v.currentTime=7">暂停看7秒</button><button onclick="v.pause();v.currentTime=11">暂停看11秒近景</button><button onclick="v.pause()">暂停</button><span id="label">本次人物粗化</span></div>
<video id="v" controls preload="metadata" poster="frame_330_1080p.png" src="plaza_coarse_people_15s_1080p.mp4"></video>
<p>1920×1080 · 30fps · 450帧 · 无音频</p>
<p class="note">本次只对人物重新计算，背景沿用上一版成片。衣物减碎纹；脸部使用较轻的简化，保留主要结构。这是软件效果预览，尚未融入FPGA模型。</p>
<p><a href="plaza_coarse_people_15s_1080p.mp4">单独打开视频</a> · <a href="people_comparison.png">查看原像素人物对照大图</a> · <a href="REPORT.md">处理与核验记录</a></p>
<img src="people_comparison.png" alt="上一版与本次人物粗化的实际同帧裁剪">
<script>const v=document.querySelector('#v'),label=document.querySelector('#label');const paths={current:'plaza_coarse_people_15s_1080p.mp4',previous:OLDPATH,source:SOURCEPATH};const names={current:'本次人物粗化',previous:'上一版对照',source:'原视频'};function swap(k){const t=v.currentTime,play=!v.paused;v.src=paths[k];label.textContent=names[k];v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration);if(play)v.play()};v.load()}</script></html>'''.replace('OLDPATH',json.dumps(old)).replace('SOURCEPATH',json.dumps(source))
    (OUT/'preview.html').write_text(html,encoding='utf-8')
    write_json(OUT/'STATUS.json',dict(status='DECODE_VERIFIED_PENDING_BROWSER',frames=450,total=450))
    files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    write_json(OUT/'manifest.json',dict(files=files,total_bytes=sum(r['bytes'] for r in files)))
    print('COARSE_PEOPLE_INSPECTED',dict(files=len(files),bytes=sum(r['bytes'] for r in files),codec=rows),flush=True)


if __name__=='__main__':main()
