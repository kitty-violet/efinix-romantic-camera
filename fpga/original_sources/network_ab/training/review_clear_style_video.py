"""Motion-compensated diagnostics and codec checks, never a visual pass score."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime import configure_workspace, write_json, file_hash
from .prepare_coherent_shared_data import pair_flow


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['teacher','guarded','student'],default='teacher');args=ap.parse_args()
    root=configure_workspace();run=root/'runs/clear-style-control-20261003';out=run/(args.model+'_videos');cv2.setNumThreads(3)
    report=json.loads((out/'report.json').read_text(encoding='utf-8'));rows=[]
    for r in report['records']:
        src=cv2.VideoCapture(r['source']);dst=cv2.VideoCapture(r['output']);prev=None;values=[];codecs=[];n=0;worst=-1;worstpair=None
        while True:
            aok,a=src.read();bok,b=dst.read();assert aok==bok
            if not aok:break
            a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
            if n in r['samples']:
                png=np.array(Image.open(out/f'{r["id"]}_{n:03d}_preencode1080.png')).astype(np.float32);err=png-b.astype(np.float32)
                codecs.append({'frame':n,'mae':float(np.abs(err).mean()),'psnr_db':float(10*np.log10(255**2/max(float(np.mean(err**2)),1e-12)))})
            smalla=cv2.resize(a,(640,360),interpolation=cv2.INTER_AREA);smallb=cv2.resize(b,(640,360),interpolation=cv2.INTER_AREA)
            if prev is not None:
                flow,mask=pair_flow(prev[0],smalla);mask=mask.astype(bool);f=flow.astype(np.float32);yy,xx=np.mgrid[:360,:640].astype(np.float32)
                style_now=smallb.astype(float)-smalla.astype(float);style_old=prev[1].astype(np.float32)-prev[0].astype(np.float32)
                warped=cv2.remap(style_old,xx+f[:,:,0],yy+f[:,:,1],cv2.INTER_LINEAR)
                value=float(np.abs(style_now-warped)[mask].mean()) if mask.any() else None
                values.append({'frame':n,'style_temporal_mae_rgb255':value,'valid_fraction':float(mask.mean())})
                if value is not None and value>worst:
                    worst=value;worstpair=np.concatenate((prev[0],smalla,prev[1],smallb),axis=1)
            prev=(smalla,smallb);n+=1
        src.release();dst.release();assert n==r['validation']['frames']
        v=np.array([z['style_temporal_mae_rgb255'] for z in values if z['style_temporal_mae_rgb255'] is not None])
        Image.fromarray(worstpair).save(out/(r['id']+'_worst_motion_pair.png'))
        rows.append({'id':r['id'],'frames':n,'median':float(np.median(v)),'p95':float(np.percentile(v,95)),
                     'worst':max(values,key=lambda x:x['style_temporal_mae_rgb255'] or -1),'encoding_samples':codecs,'per_frame':values})
        print(args.model,r['id'],'median',np.median(v),'p95',np.percentile(v,95),flush=True)
    write_json(out/'quality_review.json',{'status':'DIAGNOSTICS_NOT_ACCEPTANCE','records':rows,
       'method':'Source-derived forward/backward optical-flow consistency and RGB mask; analyze style residual at 640x360',
       'limitations':'Flow approximations, exposure and style magnitude affect scores. Low score does not prove no flicker. Native video and local inspection required.'})


if __name__=='__main__':main()
