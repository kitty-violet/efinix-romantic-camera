"""Create honest native-resolution comparisons for the WCT2 probe."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace()
    run = root / "runs/wavelet-game-probe-20261003"
    prior = root / "runs/styleid-material-probe-20261002/clean_reference720"
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 21)
    rois = {"quiet_stairs": [(120,400,520,700),(690,400,1090,700)],
            "plaza_stone": [(293,470,343,530),(778,458,828,518)],
            "park_people": [(875,305,1275,605),(0,235,175,505)]}
    stats = []
    for case, boxes in rois.items():
        dst = run / case
        shutil.copyfile(prior / case / "styleid_gamma_0.75.png", dst / "previous_styleid.png")
        names = ["source", "previous_styleid", "reconstruction", "garden_encoder"]
        labels = ["原始实拍", "上轮：叶片/人物重画", "本轮：仅还原、不加风格", "本轮：加入游戏参考"]
        images = [Image.open(dst / (name + ".png")).convert("RGB") for name in names]
        for i, box in enumerate(boxes):
            board = Image.new("RGB", (1600,365), "#131b22")
            draw = ImageDraw.Draw(board)
            original = np.asarray(images[0].crop(box)).astype(float)
            row = {"case":case,"roi":box,"mae":{}}
            for c, (name, label, image) in enumerate(zip(names, labels, images)):
                region = image.crop(box)
                row["mae"][name] = float(np.abs(np.asarray(region).astype(float)-original).mean())
                scale = "720p原尺寸局部 · 未锐化"
                if region.width < 100:
                    region = region.resize((region.width*5,region.height*5),Image.Resampling.NEAREST)
                    scale = "最近邻5倍查看 · 未补细节"
                board.paste(region,(c*400+(400-region.width)//2,65))
                draw.text((c*400+5,4),label,font=font,fill="white")
                draw.text((c*400+5,33),scale,font=font,fill="#b8c7d4")
            board.save(run / f"{case}_detail_{i}.png")
            stats.append(row)
        board = Image.new("RGB",(1280,804),"#131b22")
        draw=ImageDraw.Draw(board)
        for i,(label,im) in enumerate([(labels[0],images[0]),(labels[3],images[3])]):
            board.paste(im.resize((640,360),Image.Resampling.LANCZOS),(i*640,42))
            draw.text((i*640+8,8),label,font=font,fill="white")
        # True pixel crops below the full scene overview.
        box=boxes[0] if case=="plaza_stone" else boxes[-1 if case=="quiet_stairs" else 0]
        for i,im in enumerate((images[0],images[3])):
            crop=im.crop(box)
            if crop.width<100:crop=crop.resize((crop.width*5,crop.height*5),Image.Resampling.NEAREST)
            board.paste(crop,(i*640+(640-crop.width)//2,452))
            draw.text((i*640+8,414),"下方为原像素局部；小脸仅最近邻放大",font=font,fill="#b8c7d4")
        board.save(run/f"{case}_before_after.jpg",quality=97)
    write_json(run/"region_review.json",{"records":stats,"note":"MAE is pixel change, not a quality score"})
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>游戏风格试验 · 结构保留对照</title>
<style>body{margin:0;background:#101820;color:#edf3f8;font:17px system-ui,"Microsoft YaHei"}main{max-width:1320px;margin:auto;padding:28px}h1{font-size:30px}p{line-height:1.7;color:#c4d2de}button,select{font:inherit;padding:10px;margin:4px;background:#233744;color:white;border:1px solid #6b8798;border-radius:5px}button:focus{outline:2px solid #ffce84}video{width:100%;background:black}section{margin:26px 0;padding:18px;background:#17242d;border-radius:12px}.compare{position:relative;aspect-ratio:16/9;overflow:hidden}.compare img{position:absolute;width:100%;height:100%;object-fit:contain}.compare img:last-child{clip-path:inset(0 50% 0 0)}input[type=range]{width:100%}img.detail{width:100%;height:auto}.warning{border-left:4px solid #efb362;padding:10px 16px}a{color:#8ed8ff}small{color:#b9cad6}</style><main>
<h1>保住形体后，游戏材质感还差多少？</h1>
<p class="warning">这是本轮实际GPU推理。人物与叶片结构比上轮保留得好，但材质变化仍偏弱，天空会发灰。尚未达到最终目标，也不是新训练的FPGA权重。</p>
<section><h2>连续视频 · 10秒 / 1080p / 30fps</h2><p>台阶5秒 + 广场人物5秒。每帧经过1280×720网络计算后升至1080p。没有贴回原脸、帧插值、时间平滑或逐场景调参。</p>
<button id="showNew">看本轮结果</button><button id="showSource">看原视频</button><span id="videoLabel">本轮结果</span>
<video id="video" controls loop playsinline preload="metadata" src="video/wavelet_game_1080p30_10s.mp4"></video>
<p><a href="video/wavelet_game_1080p30_10s.mp4">单独打开本轮1080p视频</a> · <a href="video/source_1080p30_10s.mp4">单独打开原视频</a></p></section>
<section><h2>同一帧对照</h2><label>场景 <select id="scene"><option value="quiet_stairs">台阶与植物</option><option value="plaza_stone">广场人物</option><option value="park_people">逆光树丛与衣物</option></select></label>
<label>结果 <select id="variant"><option value="garden_encoder">本轮候选：保留结构</option><option value="previous_styleid">上轮StyleID：结构破坏</option><option value="reconstruction">本轮仅重建：不加风格</option><option value="sum_garden_encoder_05">失败对照：强传递产生天空色块</option><option value="garden_mid34">失败对照：只改深层几乎没有风格</option></select></label>
<p>拖动滑块：左边是原图，右边是所选结果。下面局部图可检查五官、衣领、叶片和接缝。</p>
<div class="compare"><img id="result" alt="所选结果"><img id="source" alt="原图"></div><input id="wipe" aria-label="原图覆盖比例" type="range" min="0" max="100" value="50">
<img class="detail" id="detail0" alt="原图、上轮、仅还原和本轮的局部对照"><img class="detail" id="detail1" alt="第二处局部对照">
<p id="nativeLinks"></p></section>
<section><h2>本轮使用的清晰游戏参考</h2><img class="detail" src="references/garden_720.png" alt="去掉页脚后的游戏参考"><p>来自Techland植被美术公开的《消逝的光芒：困兽》场景。这里实际使用的是场景图；灰底模型展示用于分析结构，没有整张送入迁移。</p><a href="https://izamaliszewska.artstation.com/projects/L4Yolw">查看制作人员原作</a></section>
<p>方法：WCT²作者预训练网络的FP32推理。参考图特征统计固定，内容统计按帧计算，仍需检查运动中的色调漂移。较低的重建误差不等于游戏风格已经达标。</p>
</main><script>
const scene=document.querySelector('#scene'),variant=document.querySelector('#variant'),source=document.querySelector('#source'),result=document.querySelector('#result'),wipe=document.querySelector('#wipe');
function update(){source.src=scene.value+'/source.png';result.src=scene.value+'/'+variant.value+'.png';document.querySelector('#detail0').src=scene.value+'_detail_0.png';document.querySelector('#detail1').src=scene.value+'_detail_1.png';document.querySelector('#nativeLinks').innerHTML='<a href="'+result.src+'">打开所选720p原图</a>';}
scene.addEventListener('change',update);variant.addEventListener('change',update);wipe.addEventListener('input',()=>{source.style.clipPath='inset(0 '+(100-wipe.value)+'% 0 0)'});update();
const video=document.querySelector('#video');
function choose(file,label){const time=video.currentTime,playing=!video.paused;video.src='video/'+file;document.querySelector('#videoLabel').textContent=label;video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(time,video.duration-.04);if(playing)video.play();},{once:true});}
document.querySelector('#showNew').onclick=()=>choose('wavelet_game_1080p30_10s.mp4','本轮结果');document.querySelector('#showSource').onclick=()=>choose('source_1080p30_10s.mp4','原视频');
</script></html>'''
    (run/"index.html").write_text(html,encoding="utf-8")
    print("Gallery and six native-region comparisons ready",flush=True)


if __name__=="__main__":main()
