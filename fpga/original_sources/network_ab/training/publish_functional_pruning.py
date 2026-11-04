"""Inspect and render real before/after functional-pruning outputs."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .audit_style_branches import RUN,load,REPAIR,REPAIR_SHA
from .functional_pruned_model import build
from .prune_functional_channels import data,measure,compare
from .prune_animation_teacher import profile
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .fpga_video_resize import area_3to2,bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe,video_check
from .runtime import configure_workspace,write_json,file_hash,contract_guard
from .engine import set_reproducible


def newnet(root,run):
    p=run/'recovery/functional_pruned_fp32.pt';e=json.loads((run/'recovery/export.json').read_text());assert file_hash(p)==e['sha256']
    return build(root,torch.load(p,map_location='cpu',weights_only=True)).cuda().eval()


def inspect(root,run):
    binding=json.loads((run/'recovery/binding.json').read_text());assert all(file_hash(p)==h for p,h in binding['code'].items())
    logs=[json.loads(line) for line in (run/'recovery/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,241))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 and r['native_frames']==2 for r in logs)
    assert json.loads((run/'recovery/launch_000120.json').read_text())['step']==120
    write_json(run/'recovery/training_verification.json',dict(status='PASS',steps=240,independent_process_resume_at=120,
        optimizer_seconds=sum(r['seconds'] for r in logs),peak_mib=max(r['peak_mib'] for r in logs),training_sources=len(set(r['source'] for r in logs)),
        frozen_code_hashes_match=True,full720_frames=480,preflight=json.loads((run/'recovery/preflight/result.json').read_text())['restore_error']))
    assets,flow=data(run,True);folder=run/'final_evaluation';folder.mkdir(exist_ok=False)
    net=newnet(root,run);values=measure(net,assets,flow,folder)
    before=load(root,'repair');baseline=measure(before,assets,flow);checked=compare(values,baseline)
    verification=[]
    with torch.inference_mode():
        for row,s,t in assets:
            if row['split']!='seen_regression':continue
            x=as_tensor(s)[None].cuda();p=to_rgb(net(x)[0]);stored=read_image(str(run/'recovery/validation_000240'/(row['id']+'.png')))
            error=int(np.abs(p.astype(int)-stored.astype(int)).max());assert error==0;verification.append(error)
    prof=profile(net,x);write_json(folder/'measurements.json',dict(comparison=checked,values=values,baseline=baseline,profile=prof,independent_reload_six_errors=verification))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);canvas=Image.new('RGB',(1280,944),'#151b24');draw=ImageDraw.Draw(canvas)
    labels=['原片','教师目标','修正目标续训版','按功能删减＋修正240步']
    for i,label in enumerate(labels):draw.text((320*i+6,8),label,fill='white',font=font)
    for j,(key,xy,label) in enumerate([('val_stairs_090',(424,320),'石柱'),('val_stairs_090',(180,430),'植物'),('val_plaza_090',(80,448),'人物')]):
        row=next(r for r,s,t in assets if r['id']==key);x,y=xy;top=38+290*j;draw.text((6,top),label,font=font,fill='#b9d5ef')
        paths=[Path(row['source']),Path(row['target']),root/'runs/style-retention-repair-20261003/repair/validation_000240'/(key+'.png'),folder/(key+'.png')]
        for i,p in enumerate(paths):canvas.paste(Image.open(p).crop((x,y,x+320,y+256)),(320*i,top+27))
    draw.text((6,916),'原生720像素裁切；输出没有模糊滤镜、锐化、混原片或贴脸。',font=font,fill='#b9d5ef');canvas.save(run/'final_comparison.png')
    print('FINAL_STATIC',json.dumps(checked),prof['parameters'],prof['conv_macs'],flush=True)


def render(root,run):
    nets={'new':newnet(root,run),'previous':load(root,'repair')}
    folders={'new':run/'videos','previous':run/'reference/videos'}
    for f in folders.values():f.mkdir(parents=True,exist_ok=False)
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'];records={k:[] for k in nets}
    for clip in clips:
        assert file_hash(clip['clip'])==clip['clip_sha256'];cap=cv2.VideoCapture(clip['clip'])
        paths={k:folders[k]/(clip['id']+'_1080p.mp4') for k in nets};pipes={k:encode_pipe(p,1920,1080,30) for k,p in paths.items()}
        n=0;start=time.perf_counter();last={};duplicates={k:0 for k in nets}
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert bgr.shape==(1080,1920,3);rgb=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));x=as_tensor(rgb)[None].cuda()
            for k,net in nets.items():
                with torch.inference_mode():native=to_rgb(net(x)[0])
                full=bilinear_2to3(native);payload=full.tobytes();duplicates[k]+=int(last.get(k)==payload);last[k]=payload;pipes[k].stdin.write(payload)
                if n in (90,210,359):
                    Image.fromarray(native).save(folders[k]/f'{clip["id"]}_{n:03d}_native720.png')
                    Image.fromarray(full).save(folders[k]/f'{clip["id"]}_{n:03d}_preencode1080.png')
            n+=1
            if n%90==0:print('PAIRED_VIDEO',clip['id'],n,round(time.perf_counter()-start,1),flush=True)
        cap.release();assert n==clip['validation']['frames']
        for k,pipe in pipes.items():
            close_pipe(pipe);records[k].append(dict(id=clip['id'],path=str(paths[k]),source=clip['clip'],source_sha256=clip['clip_sha256'],validation=video_check(paths[k],n),model_calls=n,identical_adjacent_outputs=duplicates[k]))
            write_json(folders[k]/'report.json',dict(status='COMPLETE' if len(records[k])==3 else 'RUNNING',records=records[k],
                weight_sha256=file_hash(run/'recovery/functional_pruned_fp32.pt') if k=='new' else REPAIR_SHA,
                pipeline='1080 integer area -> native720 FP32 network ->1080 integer bilinear',temporal_blend=False,source_blend=False,sharpen=False,crf=16))
    print('PAIRED_VIDEOS_COMPLETE',flush=True)


def diagnostics(root,run):
    from . import publish_fullres_c8_stylepush as old
    old.PREVIOUS=RUN+'/reference';old.diagnostics(root,run)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inspect','render','diagnostics']);args=ap.parse_args()
    root=configure_workspace();contract_guard();set_reproducible(2026100404);torch.set_num_threads(3);cv2.setNumThreads(2)
    globals()[args.action](root,root/'runs'/RUN)


if __name__=='__main__':main()
