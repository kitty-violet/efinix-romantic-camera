"""Short native720 video-pair diagnostic, executed from actual B1 bytes.

This is supplementary to the required full10-20s videos, not a replacement.
The source-motion mask is identical for all models and independent of targets.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .fpga_video_resize import area_3to2
from .publish_region_classes import motion_map
from .render_b1_region_trial import region_temporal_diagnostics


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=ap.parse_args()
    root=configure_workspace();set_reproducible(20261006169);cv2.setNumThreads(2);torch.set_num_threads(2)
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'];pairs=[]
    for clip in clips:
        if clip['id'] not in ('stairs','plaza'):continue
        assert file_hash(clip['clip'])==clip['clip_sha256']
        cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
        for frame in (90,180,270):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
            old=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));ok,bgr=cap.read();assert ok
            new=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));mx,my,valid=motion_map(old,new)
            pairs.append((clip['id'],frame,old,new,mx,my,valid))
        cap.release()
    results=[]
    for checkpoint in a.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=m.t.r.initialize() if a.device=='cuda' else m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
        net.load_state_dict(state['model']);net.quantized=True
        net.package['residual_encoding']=state['residual_encoding'];raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=a.device);del net
        records=[]
        for scene,frame,old,new,mx,my,valid in pairs:
            outputs=[]
            for rgb in (old,new):
                x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device=a.device,dtype=torch.float32)-128
                out=actual(x)[0].permute(1,2,0).cpu().numpy()+128;outputs.append(out)
            values=np.abs(outputs[1]-cv2.remap(outputs[0],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)).mean(2)[valid]
            records.append(dict(scene=scene,frame=frame,mean=float(values.mean()),p95=float(np.percentile(values,95)),over16=float((values>16).mean()),pixels=int(valid.sum()),
                regions=region_temporal_diagnostics(outputs[0],outputs[1],mx,my,valid)))
        mean={k:float(np.mean([r[k] for r in records])) for k in ('mean','p95','over16')}
        results.append(dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),records=records,mean=mean))
        print('REGION_TEMPORAL',checkpoint,mean,flush=True);del actual;torch.cuda.empty_cache()
    write_json(a.output,dict(results=results,actual_bytes=True,source_based_mask=True,postfilter=False,device=a.device,
        scope='6 adjacent pairs from reviewed validation clips; not full video acceptance',visual_acceptance=False))


if __name__=='__main__':main()
