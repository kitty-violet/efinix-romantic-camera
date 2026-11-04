"""Reuse four exact regression clips and freeze an additional night holdout."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from .media import decode_frames,probe_video
from .render_temporal_native_detail import encode_pipe
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=Path(r'E:\fpga\model-training-workspace\runs\coherent-shared-20261002')
    previous=json.loads((old/'evaluation_sources/clips.json').read_text(encoding='utf-8'))
    manifest=old/'data/manifest.json';data=json.loads(manifest.read_text(encoding='utf-8'))
    if file_hash(manifest)!=previous['training_manifest_sha256']:raise ValueError('Data drift')
    records=[]
    for row in previous['clips']:
        if file_hash(row['clip'])!=row['clip_sha256']:raise ValueError('Source clip drift')
        records.append({**row,'review_role':'regression_previously_seen'})
    source=next(Path(s) for s,m in data['source_splits'].items()
                if Path(s).name=='VID_20260926_192624.mp4' and m['split']=='test')
    meta=data['source_splits'][str(source)]
    if file_hash(source)!=meta['sha256']:raise ValueError('Night source drift')
    probe=probe_video(source);first=round(probe['fps']);indices=list(range(first,first+150))
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'test_night_source_1080p.mp4'
    if path.exists():raise ValueError('Evaluation source already prepared')
    pipe=encode_pipe(path,1920,1080,30);pts=[]
    board=Image.new('RGB',(1920,380),'#111');draw=ImageDraw.Draw(board)
    try:
        for n,(index,t,pixels) in enumerate(decode_frames(source,indices=indices,hlg_exposure_stops=-1.)):
            pipe.stdin.write(pixels.tobytes());pts.append(t)
            if n in (0,75,149):
                col=(0,75,149).index(n);im=Image.fromarray(pixels)
                im.save(a.output/f'test_night_{n:03d}.jpg',quality=95)
                board.paste(im.resize((640,360)),(col*640,20))
                draw.text((col*640+3,3),f'UNSEEN NIGHT SOURCE {n}/150',fill='white')
    finally:pipe.stdin.close()
    errors=pipe.stderr.read().decode('utf-8',errors='replace')
    if pipe.wait(timeout=120) or len(pts)!=150:raise RuntimeError(errors)
    board.save(a.output/'night_source_review.jpg',quality=95)
    records.append({'id':'test_night','source':str(source),'source_sha256':meta['sha256'],
        'split':'test','review_role':'additional_unseen_night_holdout',
        'clip':str(path),'clip_sha256':file_hash(path),'source_frame_indices':[first,first+149],
        'source_pts_first_last':[pts[0],pts[-1]],'frames':150,'fps':30,'size':[1920,1080],
        'preprocessing':'fixed HLG-to-SDR -1EV shared with training; no per-clip color fitting'})
    write_json(a.output/'clips.json',{'schema':'shared-model-evaluation-clips-v1',
        'status':'SOURCE_VISUAL_REVIEW_REQUIRED','training_manifest_sha256':file_hash(manifest),
        'clips':records,'previous_render':str(old/'final-2500/render.json')})
    print(a.output/'clips.json',flush=True)


if __name__=='__main__':main()
