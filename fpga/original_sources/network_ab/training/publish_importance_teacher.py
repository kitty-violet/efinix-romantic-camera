"""Publish actual weights/videos of the importance-pruned model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from . import publish_pruned_teacher as shared
from .importance_pruned_teacher import RUN,build_group_pruned
from .prune_animation_teacher import profile
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .runtime import configure_workspace,write_json,file_hash,contract_guard


def load(root,run,candidate):
    out=run/(candidate+'_recovery');report=json.loads((out/'export.json').read_text())
    p=out/'pruned_teacher_fp32.pt';assert file_hash(p)==report['sha256']
    state=torch.load(p,map_location='cpu',weights_only=True);net=build_group_pruned(root,state['indices']).cuda()
    net.load_state_dict(state['generator']);return net.eval()


def verify(root,run,candidate):
    folder=run/(candidate+'_recovery');binding=json.loads((folder/'binding.json').read_text())
    assert all(file_hash(p)==h for p,h in binding['code'].items())
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in rows]==list(range(1,601)) and all(np.isfinite(r['loss']) and r['grad_norm']>0 and r['native_frames']==2 for r in rows)
    counts=Counter(r['source'] for r in rows);assert len(counts)==7
    assert json.loads((folder/'launch_000300.json').read_text())['step']==300
    net=load(root,run,candidate);manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));errors=[]
    assert all(p.dtype==torch.float32 for p in net.parameters())
    (folder/'inference_graph.txt').write_text(str(net.generator.graph),encoding='utf-8')
    write_json(folder/'operator_inventory.json',dict(
        modules=dict(Counter(type(m).__name__ for m in net.generator.modules() if not list(m.children()))),
        input=dict(shape=[1,3,720,1280],dtype='float32',rgb_transform='uint8 / 128 - 1'),
        output=dict(shape=[1,3,720,1280],dtype='float32',rgb_transform='clip(round(tensor * 128 + 128), 0, 255).astype(uint8)'),
        note='Complete graph and pretrained factory dependencies must accompany these weights. Not the B1 graph or coefficient ABI.'))
    with torch.inference_mode():
        for row in manifest['validation']:
            x=as_tensor(read_image(row['source']))[None].cuda();p=to_rgb(net(x)[0]);saved=read_image(str(folder/'validation_000600'/(row['id']+'.png')))
            errors.append(int(np.abs(p.astype(int)-saved.astype(int)).max()))
    assert max(errors)==0
    write_json(folder/'verification.json',dict(status='FP32_RELOAD_PASS_NOT_HARDWARE_ACCEPTANCE',steps=600,
        native720_training_frames=1200,source_counts=dict(counts),optimizer_seconds=sum(r['seconds'] for r in rows),
        peak_mib=max(r['peak_mib'] for r in rows),reload_image_error=max(errors),profile=profile(net,x),
        weights_sha256=file_hash(folder/'pruned_teacher_fp32.pt')))
    print('VERIFY_PASS',candidate,flush=True)


def diagnostics(root,run,candidate):
    from . import publish_fullres_c8_stylepush as old
    # Reuse identical dense temporal measurement. Remap the parent's video paths,
    # avoiding a large video copy or any changes in the previous run.
    evaluation=run/('evaluation_'+candidate);(evaluation/'videos').mkdir(parents=True,exist_ok=False)
    write_json(evaluation/'videos/report.json',json.loads((run/('videos_'+candidate)/'report.json').read_text()))
    previous=root/'runs/teacher-pruning-calibrated-20261003/videos_keep50/report.json'
    compare=run/'parent_reference';(compare/'videos').mkdir(parents=True,exist_ok=True)
    write_json(compare/'videos/report.json',json.loads(previous.read_text()))
    old.PREVIOUS=RUN+'/parent_reference';old.diagnostics(root,evaluation)


def page(root,run,candidate):
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    panel=Image.new('RGB',(960,946),'#171c24');draw=ImageDraw.Draw(panel)
    for i,label in enumerate(['原教师目标','上一轮半通道','按重要性继续剪枝']):draw.text((320*i+7,8),label,font=font,fill='white')
    for j,(key,xy,label) in enumerate([('val_stairs_090',(424,320),'石柱'),('val_stairs_090',(180,430),'植物'),('val_plaza_090',(80,448),'人物')]):
        row=next(r for r in manifest['validation'] if r['id']==key)
        paths=[Path(row['target']),root/'runs/teacher-pruning-calibrated-20261003/keep50_recovery/validation_000300'/f'{key}.png',run/(candidate+'_recovery')/'validation_000600'/f'{key}.png']
        y=42+j*288;draw.text((8,y),label,font=font,fill='#bcd2eb');x0,y0=xy
        for i,p in enumerate(paths):panel.paste(Image.open(p).crop((x0,y0,x0+320,y0+256)),(i*320,y+28))
    draw.text((8,918),'同帧原生720像素裁切；无锐化、源片混合、贴脸或时间混帧。',font=font,fill='#bcd2eb')
    panel.save(run/'comparison.png')
    # Copy only the proven page template, then replace its observed artifact paths.
    html=(root/'runs/teacher-pruning-calibrated-20261003/index_keep50.html').read_text(encoding='utf-8')
    html=html.replace('从教师实际删除通道，再恢复训练','按实际重要性，继续压缩教师网络')
    html=html.replace('keep50：恢复300步','SELECTED：恢复600步').replace('videos_keep50/','videos_SELECTED/')
    html=html.replace('keep50_recovery/','SELECTED_recovery/').replace('evaluation_keep50/','evaluation_SELECTED/')
    html=html.replace('keep50_comparison.png','comparison.png')
    html=html.replace('教师剪枝恢复版','本轮按重要性剪枝版').replace('之前C8','上一轮半通道版')
    html=html.replace('../fullres-c8-stylepush-color-20261003/videos/','../teacher-pruning-calibrated-20261003/videos_keep50/').replace("id+'_c8_1080p.mp4'","id+'_pruned_1080p.mp4'")
    html=html.replace('原教师 / 上一轮半通道版 / 教师剪枝恢复版','原教师 / 上一轮半通道版 / 本轮按重要性剪枝版')
    html=html.replace('<p><a href="index_keep25.html">进一步压缩到四分之一通道的静态对照</a></p>', '<p><a href="importance_summary.html">查看哪些连接组和通道更重要</a></p>')
    # Face panel belongs to the parent and must not be presented as this run.
    start=html.find('<p>广场11.97秒的近处人物：')
    if start>=0:html=html[:start]+html[html.index('<script>',start):]
    if (run/'near_faces.png').exists():
        html=html.replace('<script>','<p>广场11.97秒的近处人物：原片／原教师／上一轮／本轮。同帧解码，点击原尺寸查看。</p><a href="near_faces.png"><img src="near_faces.png" alt="四版同帧近处人物"></a><script>')
    html=html.replace('<label>场景', '<p>本轮123,394参数，13.92G卷积MAC；上一轮263,264参数、28.16G。两档仍是浮点实验网络。已见静态图参与档位选择，视频用于回归检查。</p><label>场景')
    html=html.replace('<a href="RESULTS.md">结果与限制</a>', '<a href="RESULTS.md">结果与限制</a> · <a href="INFERENCE.md">权重与网络使用说明</a>')
    # Keep the requested timestamp even if the next variant is selected while
    # the preceding video's metadata is still loading.
    html=html.replace("function load(keep){let t=keep?v.currentTime:0,playing=!v.paused,id=scene.value,k=version.value;v.pause();", "let loading=false,pendingTime=0,resumeAfterLoad=false;function load(keep){let t=keep?(loading?pendingTime:v.currentTime):0,playing=loading?resumeAfterLoad:!v.paused,id=scene.value,k=version.value;pendingTime=t;resumeAfterLoad=playing;loading=true;document.querySelector('#seek').disabled=true;document.querySelector('#play').disabled=true;v.pause();")
    html=html.replace("v.currentTime=Math.min(t,v.duration-.1);if(playing)", "v.currentTime=Math.min(t,v.duration-.1);loading=false;document.querySelector('#seek').disabled=false;document.querySelector('#play').disabled=false;if(playing)")
    (run/'index.html').write_text(html.replace('SELECTED',candidate),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','render','diagnostics','page']);ap.add_argument('--candidate',choices=['cost18g','cost14g','cost10g'],required=True);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='render':shared.load=load;shared.render(root,run,args.candidate)
    else:globals()[args.mode](root,run,args.candidate)


if __name__=='__main__':main()
