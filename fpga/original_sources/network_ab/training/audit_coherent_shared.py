"""Audit full timelines without using temporal metrics as a style-quality verdict."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .runtime import file_hash,write_json


def highpass(gray):
    gray=gray.astype(np.float32)
    return gray-cv2.GaussianBlur(gray,(0,0),1.2)


def summarize(values):
    return {'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),
            'max':float(max(values))}


def audit(row):
    src,dst=Path(row['clip']),Path(row['output'])
    if file_hash(src)!=row['clip_sha256'] or file_hash(dst)!=row['output_sha256']:
        raise ValueError('Video drift')
    ca,cb=cv2.VideoCapture(str(src)),cv2.VideoCapture(str(dst))
    for cap in (ca,cb):
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS))!=(1920,1080,30.):raise ValueError('Wrong video format')
    yy,xx=np.mgrid[:360,:640].astype(np.float32)
    lasta=lastb=last_full=None;count=0;dupes=0
    vals={k:[] for k in ('pixel_change','new_white_fraction','gradient_correlation',
             'source_motion_error','styled_motion_error','source_highpass_motion_error',
             'styled_highpass_motion_error','appearance_change_motion_error')}
    encode=[]
    while True:
        oka,a=ca.read();okb,b=cb.read()
        if oka!=okb:raise ValueError('Timelines differ')
        if not oka:break
        if last_full is not None and np.array_equal(last_full,b):dupes+=1
        last_full=b
        if count in (0,75,149):
            before=cv2.imread(str(dst.parent/f"{row['id']}_{count:03d}_styled.png"))
            encode.append({'frame':count,'psnr_db':float(cv2.PSNR(before,b)),
                           'mean_absolute_error':float(np.abs(before.astype(np.float32)-b).mean())})
        a720=cv2.resize(a,(1280,720),interpolation=cv2.INTER_AREA)
        b720=cv2.resize(b,(1280,720),interpolation=cv2.INTER_AREA)
        vals['pixel_change'].append(float(np.abs(a720.astype(np.float32)-b720).mean()))
        ha,hb=cv2.cvtColor(a720,cv2.COLOR_BGR2HSV),cv2.cvtColor(b720,cv2.COLOR_BGR2HSV)
        wa=(ha[:,:,2]>=235)&(ha[:,:,1]<70);wb=(hb[:,:,2]>=235)&(hb[:,:,1]<70)
        vals['new_white_fraction'].append(float((wb&~wa).mean()))
        ga,gb=cv2.cvtColor(a720,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b720,cv2.COLOR_BGR2GRAY)
        # Same-frame structures at the actual processing resolution.
        ea=cv2.Sobel(ga,cv2.CV_32F,1,0);eb=cv2.Sobel(gb,cv2.CV_32F,1,0)
        vals['gradient_correlation'].append(float(np.corrcoef(ea.ravel()[::8],eb.ravel()[::8])[0,1]))
        a=cv2.resize(a,(640,360),interpolation=cv2.INTER_AREA).astype(np.float32)
        b=cv2.resize(b,(640,360),interpolation=cv2.INTER_AREA).astype(np.float32)
        if lasta is not None:
            flow=cv2.calcOpticalFlowFarneback(cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(lasta.astype(np.uint8),cv2.COLOR_BGR2GRAY),None,.5,4,21,4,7,1.5,0)
            mx,my=xx+flow[:,:,0],yy+flow[:,:,1]
            warp=lambda im:cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            prev_a,prev_b=warp(lasta),warp(lastb)
            source_error=np.abs(a-prev_a).mean(2)
            mask=(source_error<20)&(mx>=2)&(mx<638)&(my>=2)&(my<358)
            if mask.mean()<.15:raise ValueError('Too few motion correspondences')
            vals['source_motion_error'].append(float(source_error[mask].mean()))
            vals['styled_motion_error'].append(float(np.abs(b-prev_b).mean(2)[mask].mean()))
            ag,bg=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
            ap,bp=cv2.cvtColor(lasta,cv2.COLOR_BGR2GRAY),cv2.cvtColor(lastb,cv2.COLOR_BGR2GRAY)
            vals['source_highpass_motion_error'].append(float(np.abs(highpass(ag)-warp(highpass(ap)))[mask].mean()))
            vals['styled_highpass_motion_error'].append(float(np.abs(highpass(bg)-warp(highpass(bp)))[mask].mean()))
            vals['appearance_change_motion_error'].append(float(np.abs((b-a)-(prev_b-prev_a)).mean(2)[mask].mean()))
        lasta,lastb=a,b;count+=1
    ca.release();cb.release()
    if count!=150 or dupes:raise ValueError(f'Timeline failure {count} {dupes}')
    return {'id':row['id'],'split':row['split'],'frames':count,'fps':30,'size':[1920,1080],
            'identical_adjacent_frames':dupes,'metrics':{k:summarize(v) for k,v in vals.items()},
            'per_frame_metrics':vals,
            'timeline_note':'Appearance/spatial metrics start at frame 0; motion metrics start at frame 1.',
            'encoding_samples':encode,'approx_mbps':dst.stat().st_size*8/5/1e6,
            'output_sha256':row['output_sha256'],'weights_sha256':row['weights_sha256']}


def main():
    cv2.setNumThreads(4)
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.render.read_text(encoding='utf-8'))
    records=[]
    for row in data['clips']:
        if row['weights_sha256']!=data['weights_sha256']:raise ValueError('Per-scene model switch')
        result=audit(row);records.append(result)
        print(json.dumps({'id':result['id'],'metrics':result['metrics']},ensure_ascii=False),flush=True)
    combined=Path(data['combined_video']);cap=cv2.VideoCapture(str(combined));n=0
    while cap.read()[0]:n+=1
    cap.release()
    if n!=len(records)*150 or file_hash(combined)!=data['combined_sha256']:raise ValueError('Combined timeline mismatch')
    write_json(a.render.parent/'audit.json',{'schema':'shared-appearance-video-audit-v1',
        'status':'TECHNICAL_CHECKS_COMPLETE_VISUAL_STYLE_REVIEW_REQUIRED',
        'render_sha256':file_hash(a.render),'clips':records,'combined_frames':n,
        'interpretation':'Metrics describe geometry and temporal residuals, not proof of game-style quality, zero flicker, universal generalization, or FPGA performance.'})


if __name__=='__main__':main()
