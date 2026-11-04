"""Compare encoded reference/candidate videos with one source-motion mask.

This measures corresponding colours after approximate optical-flow alignment;
it does not establish perceptual video acceptance or perfect flow accuracy.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .fpga_video_resize import area_3to2
from .publish_region_classes import motion_map
from .runtime import configure_workspace,write_json,file_hash


def read_pair(cap,frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    result=[]
    for _ in range(2):
        ok,bgr=cap.read();assert ok and bgr.shape==(1080,1920,3)
        result.append(area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    root=configure_workspace();cv2.setNumThreads(2)
    ref=root/'runs/region-detail-deploy-20261004/coarse-people-video-20261005-v2'
    manifest=json.loads((ref/'pipeline.json').read_text(encoding='utf-8'))
    src=manifest['source_clip'];assert file_hash(src['clip'])==src['clip_sha256']
    paths=dict(source=Path(src['clip']),reference=ref/'plaza_coarse_people_15s_1080p.mp4',candidate=Path(a.candidate))
    caps={k:cv2.VideoCapture(str(v),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]) for k,v in paths.items()}
    for cap in caps.values():assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT)==450 and cap.get(cv2.CAP_PROP_FPS)==30
    records=[]
    try:
        for frame in (30,60,90,120,150,180,210,240,270,300,330,360,390,420):
            pairs={k:read_pair(cap,frame) for k,cap in caps.items()}
            mx,my,valid=motion_map(*pairs['source']);record=dict(frame=frame,valid_pixels=int(valid.sum()))
            for key,(previous,current) in pairs.items():
                warped=cv2.remap(previous.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                values=np.abs(current.astype(np.float32)-warped).mean(2)[valid]
                record[key]=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)),over16=float((values>16).mean()))
            records.append(record)
    finally:
        for cap in caps.values():cap.release()
    mean={k:{q:float(np.mean([r[k][q] for r in records])) for q in ('mean','p95','over16')} for k in caps}
    write_json(a.output,dict(paths={k:dict(path=str(p),sha256=file_hash(p)) for k,p in paths.items()},records=records,mean=mean,
        source_sha256=src['clip_sha256'],source_shared_timeline=True,all_inputs_encoded=True,
        source_mask_note='source over16 is zero by mask construction; never use it as an improvement ratio',
        scope='14 adjacent validation pairs; approximate alignment and encoded colour differences, not a perceptual flicker pass',
        reference_is_software_pipeline=True,reference_is_current_minimum_area_target=False,visual_acceptance=False))
    print('ENCODED_MOTION',mean,flush=True)


if __name__=='__main__':main()
