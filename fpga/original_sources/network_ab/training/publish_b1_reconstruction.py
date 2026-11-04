"""Review original/FP32/actual integer reconstruction, never a style demo."""
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from . import train_b1_reconstruction as r
from . import train_b1_learned_colour as c
from .b1_trained_image import decode,runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .engine import set_reproducible
from .runtime import configure_workspace,file_hash,write_json


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(r.SEED)
    train,val=r.data();out=r.OUT/'review';out.mkdir(exist_ok=False)
    fp=r.initialize();fs=torch.load(r.OUT/'fp32/best.pt',map_location='cpu',weights_only=True);fp.load_state_dict(fs['model'])
    initial=r.initialize()
    package=r.OUT/'package/coefficients-padded.bin';raw=package.read_bytes();integer=IntegerB1(runtime_from_image(raw))
    qs=torch.load(r.OUT/'qat/best.pt',map_location='cpu',weights_only=True)
    checks={};arrays={};records=[]
    for row in val:
        a=r.read(row['path']);x=r.tensor(a)[None]
        p=fp(x)[0];q=integer(x*128)/128;zero=initial(x)[0]
        record=dict(id=row['id'],initial=r.metrics(zero,x),fp32=r.metrics(p,x),integer=r.metrics(q,x),
            fp32_int8_mae=float((p-q).abs().mean()*128))
        records.append(record)
        arrays[row['id']]=[Image.fromarray(a),c.rgb_image(zero[0]),c.rgb_image(p[0]),c.rgb_image(q[0])]
        for kind,im in zip(['source720','initial','fp32','integer'],arrays[row['id']]):
            im.save(out/(row['id']+'_'+kind+'.png'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1680,314*len(c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(c.CROPS):
        for j,(im,title) in enumerate(zip(arrays[key],['原图','训练前','FP32重建','实际INT8包重建'])):
            board.paste(im.crop(box),(420*j,314*i+34))
            draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(out/'four_way_native_crops.png')
    # Decode strict transport and verify malformed inputs fail rather than load.
    decode(raw)
    for label,bad in [('truncated',raw[:-1]),('crc',raw[:1200]+bytes([raw[1200]^1])+raw[1201:]),('padding',raw[:-1]+b'\x01')]:
        try:decode(bad)
        except (ValueError,AssertionError):checks[label]='rejected'
        else:raise AssertionError('Accepted invalid package '+label)
    hexdata=b''.join(int(line,16).to_bytes(8,'little') for line in (r.OUT/'package/load_words.hex').read_text().splitlines())
    assert hexdata==raw;checks['hex_roundtrip']='PASS'
    pooled={}
    for stage in ('initial','fp32','integer'):
        mean=sum(v[stage]['mse_rgb255'] for v in records)/len(records)
        pooled[stage]=dict(mse=mean,psnr=10*math.log10(255**2/mean),mae=sum(v[stage]['mae'] for v in records)/len(records))
    fpdone=json.loads((r.OUT/'fp32/completion.json').read_text());qdone=json.loads((r.OUT/'qat/completion.json').read_text())
    write_json(out/'results.json',dict(records=records,pooled=pooled,package_checks=checks,
        fp32_best_step=fs['step'],qat_best_step=qs['step'],fp32=fpdone,qat=qdone,
        training_images=len(train),validation_images=len(val),heldout_source_not_blind=True,
        weights=str(package),sha256=file_hash(package),fixed_contract_sha256=r.CONTRACT_SHA,
        no_hardware_changes=True,no_style_objective=True,no_external_postprocessing=True,
        theoretical_best_proven=False))
    sections=[]
    for key,title in [('val_stairs_090','台阶与草丛'),('val_plaza_210','人物与树木'),('val_new_hallway_120','走廊')]:
        sections.append(f'<h2>{title} · 原生720p</h2><div class="pair"><figure><figcaption>原图／训练目标</figcaption><img src="review/{key}_source720.png"></figure><figure><figcaption>实际INT8包重建</figcaption><img src="review/{key}_integer.png"></figure></div>')
    clips=json.loads((r.OUT.parent/'clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    for clip in clips:
        if clip['id'] not in ('stairs','plaza'):continue
        source=Path(os.path.relpath(clip['clip'],r.OUT)).as_posix()
        seconds=12 if clip['id']=='stairs' else 15
        sections.append(f'<h2>{clip["id"]} · {seconds}秒 · 1080p30</h2><div class="pair videos"><figure><figcaption>原片</figcaption><video controls preload="metadata" src="{html.escape(source)}"></video></figure><figure><figcaption>1080→720→B1重建→1080</figcaption><video controls preload="metadata" src="videos/{clip["id"]}_integer1080.mp4"></video></figure></div><button onclick="playPair(this)">同步从头播放</button>')
    page='''<!doctype html><meta charset="utf-8"><title>B1原图重建能力测试</title>
<style>body{margin:24px;background:#16202b;color:#eee;font:17px system-ui;line-height:1.65}img,video{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;min-width:0}button{padding:9px 20px}a{color:#9ed2ff}</style>
<h1>B1原图重建能力测试</h1><p>固定硬件网络，190张原图训练。没有风格目标、色层限制、外部锐化或高清旁路。图片是720p无损PNG；视频是实际20KiB系数包的软件整数复演，未上板。可点击图片查看原尺寸，视频可全屏。</p>'''+''.join(sections)+'''
<script>document.querySelectorAll('img').forEach(x=>x.onclick=()=>window.open(x.src));function playPair(b){let vs=b.previousElementSibling.querySelectorAll('video');vs.forEach(v=>{v.currentTime=0;v.play()})}</script>'''
    (r.OUT/'watch.html').write_text(page,encoding='utf-8')
    print('RECON_REVIEW_COMPLETE',json.dumps(pooled),flush=True)


if __name__=='__main__':main()
