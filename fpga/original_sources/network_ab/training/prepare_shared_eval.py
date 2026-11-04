"""Freeze full 5-second evaluation clips before seeing model outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from .media import decode_frames,probe_video
from .render_temporal_native_detail import encode_pipe
from .runtime import file_hash,write_json


CLIPS=[('train_escalator','VID_20260925_070530.mp4',8.,'train'),
       ('holdout_corridor','VID_20260925_113231.mp4',1.,'holdout_indoor'),
       ('validation_plaza','VID_20260925_101046.mp4',16.,'val'),
       ('test_street','VID_20260925_070032.mp4',0.,'test')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.manifest.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True);records=[]
    for name,filename,start,split in CLIPS:
        matches=[(p,r) for p,r in data['source_splits'].items() if Path(p).name==filename]
        if len(matches)!=1 or matches[0][1]['split']!=split:raise ValueError('Evaluation split changed')
        source,meta=matches[0]
        if file_hash(source)!=meta['sha256']:raise ValueError('Source drift')
        probe=probe_video(source);first=round(start*probe['fps']);indices=list(range(first,first+150))
        path=args.output/f'{name}_source_1080p.mp4'
        pipe=encode_pipe(path,1920,1080,30);pts=[]
        try:
            for n,(index,t,pixels) in enumerate(decode_frames(source,indices=indices,hlg_exposure_stops=-1.)):
                if pixels.shape!=(1080,1920,3):raise ValueError('Unexpected source dimensions')
                pipe.stdin.write(pixels.tobytes());pts.append(t)
                if n in (0,75,149):Image.fromarray(pixels).save(args.output/f'{name}_{n:03d}.jpg',quality=95)
        finally:pipe.stdin.close()
        error=pipe.stderr.read().decode('utf-8',errors='replace')
        if pipe.wait(timeout=120) or len(pts)!=150:raise RuntimeError(error)
        cap=cv2.VideoCapture(str(path));count=0
        while cap.read()[0]:count+=1
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if count!=150 or fps!=30:raise RuntimeError('Encoded source mismatch')
        records.append({'id':name,'source':source,'source_sha256':meta['sha256'],
            'split':split,'clip':str(path),'clip_sha256':file_hash(path),
            'source_frame_indices':[first,first+149],'source_pts_first_last':[pts[0],pts[-1]],
            'frames':150,'fps':30,'size':[1920,1080],
            'preprocessing':'fixed HLG-to-SDR -1EV shared with training; no per-clip color fitting'})
        write_json(args.output/'clips.json',{'schema':'shared-model-evaluation-clips-v1',
            'status':'SOURCE_VISUAL_REVIEW_REQUIRED','training_manifest_sha256':file_hash(args.manifest),
            'clips':records})
        print('CLIP_READY',name,flush=True)
    board=Image.new('RGB',(3*640,len(records)*380),'#111');draw=ImageDraw.Draw(board)
    for y,row in enumerate(records):
        for x,i in enumerate((0,75,149)):
            im=Image.open(args.output/f"{row['id']}_{i:03d}.jpg").resize((640,360),Image.Resampling.LANCZOS)
            board.paste(im,(x*640,y*380+20));draw.text((x*640+3,y*380+3),f"{row['id']} {i}/150",fill='white')
    board.save(args.output/'source_clips_review.jpg',quality=93)


if __name__=='__main__':main()
