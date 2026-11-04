"""Honest side-by-side delivery of algorithm, learned classes and color-only control."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .train_region_classes import ROOT,RUN,ClassNet,output,labels,as_tensor,read_image,source_manifest
from .region_mechanism_probe import tensor,rgb,l0_surface,gaussian,outline
from .runtime import configure_workspace,file_hash,write_json,contract_guard
from .engine import set_reproducible
from .fpga_video_resize import area_3to2,bilinear_2to3
from .media import decoder_identity
from .render_wavelet_game import close_pipe,video_check


def load(step=2000):
    run=ROOT/'runs'/RUN;folder=run/'classes32';state=torch.load(folder/f'checkpoint_{step:06d}.pt',map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=ClassNet().cuda().eval();net.load_state_dict(state['model'])
    palette=torch.tensor(state['palette'],device='cuda',dtype=torch.float32)/255
    return run,folder,state,net,palette


@torch.no_grad()
def algorithm(s):return rgb(outline(l0_surface(gaussian(tensor(s),.9),.015)))


def export(run,folder,state,net,palette):
    out=folder/'final';out.mkdir(exist_ok=False)
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,state['step']+1))
    payload={k:state[k] for k in ('model','step','palette','binding')};payload.update(production=False,
        inference='ClassNet -> per-pixel argmax32 -> fixed palette. No teacher/FFT/postblur/sharpen/flow/face/source blending.',
        hardware_warning='Different convolution graph AND output format. Not B1-flashable. INT8/RTL/fit/timing unverified.')
    weight=out/'region_classes_fp32.pt';torch.save(payload,weight)
    reload=ClassNet().cuda().eval();reload.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['model'])
    checks=[]
    with torch.no_grad():
        for r in source_manifest()['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();a=net(x);b=reload(x);err=float((a-b).abs().max());assert err==0
            expected=read_image(str(folder/f'validation_{state["step"]:06d}'/(r['id']+'.png')))
            pe=int(np.abs(rgb(output(reload,x,palette)).astype(int)-expected.astype(int)).max());assert pe==0
            checks.append(dict(id=r['id'],logit_max_error=err,png_max_error=pe))
    write_json(out/'verification.json',dict(status='FP32_EXPORT_RELOAD_PASS_NOT_STYLE_OR_DEPLOYMENT_ACCEPTANCE',checks=checks,
        sha256=file_hash(weight),parameters=sum(p.numel() for p in net.parameters()),step=state['step'],
        optimizer_seconds=sum(r['seconds'] for r in rows),native_updates=sum(r['native'] for r in rows),
        independent_resume_from250=(folder/'launch_000250.json').exists(),palette_colors=len(palette)))
    (out/'INFERENCE.md').write_text('加载`train_region_classes.ClassNet`，权重model字段，eval。RGB输入float32=(uint8-128)/128，NCHW、原生1280×720。输出32个logit，按像素argmax，在权重palette字段查RGB uint8。共享全图同一固定颜色表，无每帧拟合。当前FPGA没有这个类别输出接口，不能直接刷入。\n',encoding='utf-8')
    print('CLASS_EXPORT',file_hash(weight),flush=True)


def pictures(run,folder,state,net,palette):
    manifest=source_manifest();alg=run/'algorithm';ctrl=run/'palette_only_control';alg.mkdir(exist_ok=True);ctrl.mkdir(exist_ok=True)
    for r in manifest['validation']:
        s=read_image(r['source']);Image.fromarray(algorithm(s)).save(alg/(r['id']+'.png'))
        with torch.no_grad():a=labels(tensor(s),palette);control=rgb(palette[a].permute(0,3,1,2))
        Image.fromarray(control).save(ctrl/(r['id']+'.png'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);specs=[('val_stairs_090',(424,320)),('val_stairs_090',(150,430)),('val_plaza_090',(80,400))]
    board=Image.new('RGB',(1600,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate(specs):
        r=next(r for r in manifest['validation'] if r['id']==key)
        paths=[r['source'],r['target'],alg/(key+'.png'),ctrl/(key+'.png'),folder/f'validation_{state["step"]:06d}'/(key+'.png')]
        names=['原片','原教师参照','区域算法实验','只减颜色：对照',f'类别网络 {state["step"]}步']
        for i,(p,name) in enumerate(zip(paths,names)):
            d.text((320*i+5,298*j+5),name,font=font,fill='white');board.paste(Image.open(p).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(run/'comparison.png')


def encode(path):
    return subprocess.Popen([decoder_identity()['path'],'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s:v','1920x1080','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-threads','2','-crf','16',
        '-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(path)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)


def motion_map(previous,current):
    a,b=[cv2.resize(v,(640,360),interpolation=cv2.INTER_AREA) for v in (previous,current)]
    backward=cv2.calcOpticalFlowFarneback(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None,.5,3,21,3,5,1.2,0)
    flow=cv2.resize(backward,(1280,720),interpolation=cv2.INTER_LINEAR)*2
    yy,xx=np.mgrid[:720,:1280].astype(np.float32);mx,my=xx+flow[:,:,0],yy+flow[:,:,1]
    warped=cv2.remap(previous.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    mask=(np.abs(current.astype(np.float32)-warped).mean(2)<8)&(mx>=8)&(mx<1272)&(my>=8)&(my<712)
    return mx,my,mask


def render(run,folder,state,net,palette):
    clips=json.loads((ROOT/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    train={r['raw_source'] for r in json.loads((folder/'data.json').read_text(encoding='utf-8'))['records']}
    out=run/'videos';out.mkdir(exist_ok=False);report=[]
    for clip in clips:
        assert clip['source'] not in train and file_hash(clip['clip'])==clip['clip_sha256']
        keys=['network','algorithm']+(['palette_only'] if clip['id']=='stairs' else [])
        paths={k:out/(clip['id']+'_'+k+'_1080p.mp4') for k in keys};pipes={k:encode(p) for k,p in paths.items()}
        cap=cv2.VideoCapture(clip['clip']);previous=None;logs=[];n=0;start=time.perf_counter();same={k:0 for k in keys}
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                s=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    versions=dict(network=rgb(output(net,as_tensor(s)[None].cuda(),palette)),algorithm=algorithm(s))
                    if 'palette_only' in keys:versions['palette_only']=rgb(palette[labels(tensor(s),palette)].permute(0,3,1,2))
                for k,v in versions.items():
                    full=bilinear_2to3(v);pipes[k].stdin.write(full.tobytes())
                    if n in (90,120,210):
                        Image.fromarray(v).save(out/(clip['id']+f'_{k}_{n:03d}_native720.png'))
                        Image.fromarray(full).save(out/(clip['id']+f'_{k}_{n:03d}_preencode1080.png'))
                current=dict(source=s,**versions)
                if previous is not None:
                    mx,my,mask=motion_map(previous['source'],s);row=dict(frame=n,valid_fraction=float(mask.mean()))
                    for k,v in current.items():
                        w=cv2.remap(previous[k].astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        err=np.abs(v.astype(np.float32)-w).mean(2)
                        row[k]=float(err[mask].mean()) if mask.any() else None
                        if k!='source':same[k]+=int(np.array_equal(v,previous[k]))
                    logs.append(row)
                previous=current;n+=1
                if n%60==0:print('REGION_VIDEO',clip['id'],n,round(time.perf_counter()-start,2),flush=True)
        finally:
            cap.release()
            for pipe in pipes.values():close_pipe(pipe)
        assert n==clip['validation']['frames'];checks={k:video_check(p,n) for k,p in paths.items()}
        summary={k:dict(mean=float(np.mean([r[k] for r in logs if r[k] is not None])),p95=float(np.percentile([r[k] for r in logs if r[k] is not None],95))) for k in ['source']+keys}
        write_json(out/(clip['id']+'_motion.json'),dict(records=logs,summary=summary,method='Source-flow compensated native720 RGB MAE; source photometric mask<8. Diagnostic, not a visual stability pass.'))
        report.append(dict(id=clip['id'],frames=n,seconds=n/30,paths={k:str(p) for k,p in paths.items()},checks=checks,
            identical_adjacent_outputs=same,motion=summary,source=clip['clip'],source_sha256=clip['clip_sha256']))
        write_json(out/'report.json',dict(status='COMPLETE' if len(report)==len(clips) else 'RUNNING',records=report,
            native_processing=[1280,720],pipeline='1080 integer area ->720 processing ->1080 integer bilinear',
            fps=30,post_network_blur=False,post_network_sharpen=False,frame_blending=False,duplicate_padding=False,
            algorithm_is_network=False,hardware_verified=False,weights_sha256=file_hash(folder/'final/region_classes_fp32.pt')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['export','pictures','render']);a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100416);cv2.setNumThreads(2)
    values=load();globals()[a.action](*values)


if __name__=='__main__':main()
