"""Publish original/old/new integer recovery, with a reserved final test set."""
import argparse
import html
import json
import os
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_reconstruction as r
from . import research_b1_reconstruction as s
from .probe_b1_cartoon_int8 import IntegerB1
from .b1_trained_image import runtime_from_image,decode
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .audit_b1_reconstruction_research import phase_tests,activation_summary


class ActualBytes:
    def __init__(self,path):self.net=IntegerB1(runtime_from_image(Path(path).read_bytes()))
    def __call__(self,x):return self.net(x*128)/128,None


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fp32',required=True);ap.add_argument('--qat',required=True)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100565);contract_guard()
    train,val=r.data();test=json.loads((s.OUT/'holdout/manifest.json').read_text())['records']
    assert not {x['video'] for x in train}&{x['video'] for x in test}
    for row in test:assert file_hash(row['path'])==row['sha256']
    dest=s.OUT/'delivery';dest.mkdir(exist_ok=False)
    q=r.initialize();qd=torch.load(a.qat,map_location='cpu',weights_only=True);q.load_state_dict(qd['model']);q.quantized=True
    r.export_review(q,val,dest)
    shutil.copy2(a.fp32,dest/'fp32_best.pt');shutil.copy2(a.qat,dest/'qat_best.pt')
    raw=(dest/'coefficients-padded.bin').read_bytes();decode(raw)
    assert len(raw)==20480
    assert b''.join(int(x,16).to_bytes(8,'little') for x in (dest/'load_words.hex').read_text().splitlines())==raw
    for key,bad in [('length',raw[:-1]),('crc',raw[:1300]+bytes([raw[1300]^1])+raw[1301:]),('padding',raw[:-1]+b'\x01')]:
        try:decode(bad)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('Accepted invalid '+key)
    fp=r.initialize();fd=torch.load(a.fp32,map_location='cpu',weights_only=True);fp.load_state_dict(fd['model']);fp.quantized=False
    oldfp=r.initialize();oldfp.load_state_dict(torch.load(r.OUT/'fp32/best.pt',map_location='cpu',weights_only=True)['model'])
    models={'old_fp32':oldfp,'old_integer':ActualBytes(r.OUT/'delivery/coefficients-padded.bin'),
            'new_fp32':fp,'new_integer':ActualBytes(dest/'coefficients-padded.bin')}
    metrics={}
    for label,net in models.items():
        metrics[label]=dict(validation=s.evaluate(net,val),reserved_test=s.evaluate(net,test))
        print('FINAL_RESEARCH',label,metrics[label]['validation']['pooled_psnr'],metrics[label]['reserved_test']['pooled_psnr'],flush=True)
    phase={label:phase_tests(net,val)
           for label,net in models.items() if 'integer' in label}
    activations={label:activation_summary(net,val) for label,net in models.items() if 'fp32' in label}
    images={};review=s.OUT/'final-review';review.mkdir(exist_ok=False)
    for row in val:
        x=r.tensor(r.read(row['path']))[None]
        frames=[Image.fromarray(r.read(row['path']))]
        for label in ('old_integer','new_fp32','new_integer'):
            im=r.c.rgb_image(models[label](x)[0][0]);im.save(review/(row['id']+'_'+label+'.png'));frames.append(im)
        frames[0].save(review/(row['id']+'_source.png'));images[row['id']]=frames
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1260,314*len(r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(r.c.CROPS):
        for j,(im,title) in enumerate(zip([images[key][0],images[key][1],images[key][3]],['原图720p','上一版实际INT8','本轮实际INT8'])):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(review/'comparison.png')
    # All ten test inputs are listed, independent of which outputs look best.
    test_sections=[]
    for idx,row in enumerate(test):
        x=r.tensor(r.read(row['path']))[None];im=r.c.rgb_image(models['new_integer'](x)[0][0])
        path=review/(row['id']+'_new_integer.png');im.save(path)
        rel=Path(os.path.relpath(row['path'],s.OUT)).as_posix()
        test_sections.append(f'<details><summary>预留测试 {html.escape(row["id"])}</summary><div class="pair"><img src="{rel}"><img src="final-review/{path.name}"></div></details>')
    sections=[]
    for key,title in [('val_stairs_090','台阶与草丛'),('val_plaza_210','人物与树木'),('val_new_hallway_120','走廊')]:
        imgs=''.join(f'<figure><figcaption>{caption}</figcaption><a href="final-review/{key}_{tag}.png"><img src="final-review/{key}_{tag}.png"></a></figure>'
            for tag,caption in [('source','原图720p'),('old_integer','上一版实际包'),('new_integer','本轮实际包')])
        sections.append(f'<h2>{title}</h2><div class="triple">{imgs}</div>')
    clips=json.loads((r.OUT.parent/'clear-style-control-20261003/video_clips.json').read_text())['clips']
    for clip in clips:
        if clip['id'] not in ('stairs','plaza'):continue
        sources=[clip['clip'],str(r.OUT/'videos'/(clip['id']+'_integer1080.mp4')),str(s.OUT/'videos'/(clip['id']+'_integer1080.mp4'))]
        vids=''.join(f'<figure><figcaption>{caption}</figcaption><video controls preload="metadata" src="{Path(os.path.relpath(path,s.OUT)).as_posix()}"></video></figure>' for caption,path in zip(['原片','上一版重建','本轮实际INT8重建'],sources))
        sections.append(f'<h2>{clip["id"]} · {12 if clip["id"]=="stairs" else 15}秒 / 1080p30</h2><div class="triple">{vids}</div><button onclick="playPair(this)">同步从头播放</button>')
    page='''<!doctype html><meta charset="utf-8"><title>B1原图恢复研究</title><style>body{background:#15202b;color:#eee;font:17px system-ui;margin:24px;line-height:1.6}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}img,video{width:100%}button{padding:10px;margin:10px 0}a{color:#99d3ff}</style><h1>固定B1：原图恢复研究</h1><p>原图→同一个720/360/180/360/720网络→原图。图像为无损PNG，点击可看原尺寸。视频按1080→整数area720→真实20KiB权重包→整数bilinear1080生成，没有额外锐化、滤镜或高清旁路。软件整数回放，未上板。小人脸和密集纹理仍有恢复损失，不宣称无损或理论极限。</p>'''+''.join(sections)+'<h2>两个来源隔离视频的10张预留测试</h2>'+''.join(test_sections)+'''<script>function playPair(b){b.previousElementSibling.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})}</script>'''
    (s.OUT/'watch.html').write_text(page,encoding='utf-8')
    write_json(s.OUT/'final_results.json',dict(fp32_source=a.fp32,qat_source=a.qat,metrics=metrics,phase=phase,activations=activations,
        package_sha256=file_hash(dest/'coefficients-padded.bin'),contract_sha256=r.CONTRACT_SHA,
        transport_bytes=20480,core_bytes=17792,actual_board=False,external_filter=False,
        theoretical_limit_proven=False,test_used_for_selection=False,all_test_frames_reported=True))


if __name__=='__main__':main()
