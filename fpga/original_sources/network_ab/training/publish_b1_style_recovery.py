"""Export actual B1 bytes and compare style fidelity without external filters."""
import argparse
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import train_b1_style_recovery as m
from . import train_b1_reconstruction as r
from .publish_b1_reconstruction_research import ActualBytes
from .audit_b1_reconstruction_research import phase_tests
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible


@torch.inference_mode()
def export(net,rows,dest):
    dest.mkdir(exist_ok=False);assert net.quantized
    device=next(net.parameters()).device
    raw=net.export();assert len(raw)==20480
    path=dest/'coefficients-padded.bin';path.write_bytes(raw)
    words=''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n' for i in range(0,len(raw),8))
    (dest/'load_words.hex').write_text(words,encoding='ascii')
    assert b''.join(int(x,16).to_bytes(8,'little') for x in words.splitlines())==raw
    actual=r.IntegerB1(r.runtime_from_image(raw),device=str(device));records=[]
    for row in rows:
        x=torch.tensor(r.read(row['path']).transpose(2,0,1)[None].copy(),device=device,dtype=torch.float32)/128-1
        nodes,_=actual(x*128,True)
        ours=net(x,all_nodes=True)
        error=max(float((v-nodes[k]).abs().max()) for k,v in ours.items());assert error==0
        records.append(dict(id=row['id'],export_node_error=error,nodes=len(nodes)-1,input_checked=True))
    from .b1_package_oracle import oracle_records
    params,residual=r.decode(raw);op,rp=oracle_records(params,residual)
    sample=np.random.default_rng(81).integers(-128,128,(12,12,3),dtype=np.int16)
    contract,_=contract_guard();ref=r.execute_integer(contract,sample.tolist(),op,rp)
    got,_=actual(torch.tensor(sample.transpose(2,0,1)[None].copy(),device=device,dtype=torch.float32),True)
    error=max(float(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in got.items())
    assert error==0
    for key,bad in [('length',raw[:-1]),('crc',raw[:1300]+bytes([raw[1300]^1])+raw[1301:]),('padding',raw[:-1]+b'\x01')]:
        try:r.decode(bad)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('Accepted invalid '+key)
    write_json(dest/'review.json',dict(integer=True,independent_oracle_error=error,records=records,
        package_sha256=file_hash(path),purpose='style_candidate_numeric_verification',visual_acceptance=False,actual_board=False,
        rejected_invalid_length_crc_padding=True,hex_roundtrip=True,external_filter=False,residual_encoding=residual))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fp32',required=True);ap.add_argument('--qat',required=True)
    ap.add_argument('--assessment',default='画质研究候选，尚未获得用户画风验收。')
    ap.add_argument('--goal-status',choices=['not_met','candidate_review'],default='candidate_review')
    a=ap.parse_args();configure_workspace();set_reproducible(2026100587);contract_guard();d=m.data()
    fp=r.initialize();fd=torch.load(a.fp32,map_location='cpu',weights_only=True);fp.load_state_dict(fd['model']);fp.quantized=False
    q=r.initialize();qd=torch.load(a.qat,map_location='cpu',weights_only=True);q.load_state_dict(qd['model']);q.quantized=True
    dest=m.OUT/'delivery';export(q,d['validation'],dest)
    shutil.copy2(a.fp32,dest/'fp32_best.pt');shutil.copy2(a.qat,dest/'qat_best.pt')
    old=r.OUT.parent/'b1-sampling-colour-v2-20261005/nineteen-levels/crisp-edges/qat/package/coefficients-padded.bin'
    models=dict(old_integer=ActualBytes(old),new_fp32=fp,new_integer=ActualBytes(dest/'coefficients-padded.bin'))
    results={};images={row['id']:[Image.fromarray(r.read(row['path'])),Image.fromarray(r.read(row['target']))] for row in d['validation']}
    review=m.OUT/'final-review';review.mkdir(exist_ok=False)
    for label,net in models.items():
        results[label]=dict(validation=m.evaluate(net,d['validation']),phase=phase_tests(net,d['validation']))
        if label!='old_integer':results[label]['reserved_test']=m.evaluate(net,d['reserved_test'])
        for row in d['validation']:
            im=r.c.rgb_image(net(r.tensor(r.read(row['path']))[None])[0][0]);im.save(review/(row['id']+'_'+label+'.png'))
            images[row['id']].append(im)
        print('FINAL_STYLE',label,results[label]['validation']['mean'],flush=True)
    for row in d['validation']:
        images[row['id']][0].save(review/(row['id']+'_source.png'));images[row['id']][1].save(review/(row['id']+'_target.png'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for suffix,columns in [('comparison',[(0,'原图'),(1,'认可目标'),(4,'本轮实际INT8')]),
                            ('old-new',[(1,'认可目标'),(2,'旧19级实际包'),(4,'本轮实际INT8')]),
                            ('quantization',[(1,'认可目标'),(3,'清晰FP32参照'),(4,'本轮实际INT8')])]:
        board=Image.new('RGB',(1260,314*len(r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
        for i,(key,box,label) in enumerate(r.c.CROPS):
            for j,(index,title) in enumerate(columns):
                board.paste(images[key][index].crop(box),(420*j,314*i+34))
                draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
        board.save(review/(suffix+'.png'))
    test_sections=[]
    for row in d['reserved_test']:
        assert file_hash(row['path'])==row['sha256'] and file_hash(row['target'])==row['target_sha256']
        im=r.c.rgb_image(models['new_integer'](r.tensor(r.read(row['path']))[None])[0][0]);im.save(review/(row['id']+'_new_integer.png'))
        sources=[row['path'],row['target'],str(review/(row['id']+'_new_integer.png'))]
        imgs=''.join(f'<img src="{Path(os.path.relpath(p,m.OUT)).as_posix()}">' for p in sources)
        test_sections.append(f'<details><summary>{row["id"]}</summary><div class="triple">{imgs}</div></details>')
    write_json(m.OUT/'final_results.json',dict(models=results,fp32_source=a.fp32,qat_source=a.qat,
        fp32_sha256=file_hash(a.fp32),qat_sha256=file_hash(a.qat),package_sha256=file_hash(dest/'coefficients-padded.bin'),
        contract_sha256=r.CONTRACT_SHA,data_sha256=file_hash(m.OUT/'data.json'),
        core_bytes=17792,transport_bytes=20480,actual_board=False,external_filter=False,
        test_used_for_selection=False,globally_optimal=False,assessment=a.assessment,goal_status=a.goal_status))
    sections=[]
    for key,title in [('val_stairs_090','台阶与草丛'),('val_plaza_210','人物与树木'),('val_new_hallway_120','走廊')]:
        imgs=''.join(f'<figure><figcaption>{caption}</figcaption><a href="final-review/{key}_{tag}.png"><img src="final-review/{key}_{tag}.png"></a></figure>'
            for tag,caption in [('source','原图720p'),('target','认可的风格目标'),('new_integer','本轮实际20KiB包')])
        sections.append(f'<h2>{title}</h2><div class="triple">{imgs}</div>')
    clips=json.loads((r.OUT.parent/'clear-style-control-20261003/video_clips.json').read_text())['clips']
    for clip in clips:
        if clip['id'] not in ('stairs','plaza'):continue
        paths=[clip['clip'],str(m.OUT/'videos'/(clip['id']+'_integer1080.mp4'))]
        vids=''.join(f'<figure><figcaption>{title}</figcaption><video controls preload="metadata" src="{Path(os.path.relpath(path,m.OUT)).as_posix()}"></video></figure>' for path,title in zip(paths,['原片','本轮实际整数包']))
        sections.append(f'<h2>{clip["id"]} · {12 if clip["id"]=="stairs" else 15}秒</h2><div class="pair">{vids}</div><button onclick="playPair(this)">同步从头播放</button>')
    page='''<!doctype html><meta charset="utf-8"><title>B1清晰风格训练</title><style>body{background:#15202b;color:#eee;font:17px system-ui;margin:24px;line-height:1.6}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}img,video{width:100%}button{padding:10px;margin:10px 0}a{color:#99d3ff}</style><h1>固定B1：清晰风格训练</h1><p>1080→整数area720→真实20KiB系数包→整数bilinear1080。软件整数离线回放、30fps播放，未上板。右侧所有画面只经过B1，未加外部滤镜、色阶器、分割或高清旁路。中间是训练目标，并非小网络输出。点击图片查看原尺寸。</p>'''+''.join(sections)+'<h2>原尺寸裁图</h2><img src="final-review/comparison.png"><h2>与旧19级学生对比</h2><img src="final-review/old-new.png"><h2>量化前后</h2><img src="final-review/quantization.png"><h2>两个隔离来源的全部10张预留测试</h2>'+''.join(test_sections)+'''<script>function playPair(b){b.previousElementSibling.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})}</script>'''
    page=page.replace('<h2>量化前后</h2>','<h2>浮点参照与整数候选</h2>')
    page=page.replace('<h1>固定B1：清晰风格训练</h1>',
        '<h1>固定B1：清晰风格训练</h1><p><strong>'+html.escape(a.assessment)+'</strong></p>')
    (m.OUT/'watch.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
