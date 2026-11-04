"""Freeze source-level splits and prepare real adjacent-frame training pairs."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .media import decode_frames
from .runtime import configure_workspace, file_hash, write_json


def contact(rows, out, prefix, path_key, per_page=24):
    for start in range(0, len(rows), per_page):
        page = rows[start:start+per_page]
        board = Image.new("RGB", (4*320, ((len(page)+3)//4)*210), "#181818")
        draw = ImageDraw.Draw(board)
        for i, row in enumerate(page):
            image = Image.open(row[path_key]).convert("RGB")
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            x, y = (i % 4)*320, (i // 4)*210
            board.paste(image, (x, y+30))
            draw.text((x+3, y+2), f"{start+i:02d} {Path(row[path_key]).name}", fill="white")
            draw.text((x+3, y+15), row.get("scene",row.get("split","")), fill="white")
        board.save(out / f"{prefix}_{start//per_page:02d}.jpg", quality=92)


def pair_flow(a, b):
    a = cv2.resize(a, (640, 360), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (640, 360), interpolation=cv2.INTER_AREA)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    backward = cv2.calcOpticalFlowFarneback(gb, ga, None, .5, 4, 21, 4, 7, 1.5, 0)
    forward = cv2.calcOpticalFlowFarneback(ga, gb, None, .5, 4, 21, 4, 7, 1.5, 0)
    yy, xx = np.mgrid[:360, :640].astype(np.float32)
    mx, my = xx+backward[:,:,0], yy+backward[:,:,1]
    warped = cv2.remap(a, mx, my, cv2.INTER_LINEAR)
    fwarped = cv2.remap(forward, mx, my, cv2.INTER_LINEAR)
    fb = np.linalg.norm(backward+fwarped, axis=2)
    error = np.abs(warped.astype(np.float32)-b.astype(np.float32)).mean(2)
    mask = (mx>=1)&(mx<638)&(my>=1)&(my<358)&(fb<1.5)&(error<18)
    return backward.astype(np.float16), mask.astype(np.uint8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pairs",action="store_true")
    args=parser.parse_args()
    root=configure_workspace()
    content=root/'data/processed/content-20261001-v4-minus1ev-draft/manifest.approved.json'
    target=root/'data/processed/style-target-20261001-v14-expanded-reviewed/dataset-manifest.json'
    c=json.loads(content.read_text(encoding='utf-8'))
    s=json.loads(target.read_text(encoding='utf-8'))
    # Freeze an additional unseen indoor location before any new training.
    for row in c['images']:
        if Path(row['source']).name=='VID_20260925_113231.mp4':
            row['split']='holdout_indoor'
    args.output.mkdir(parents=True,exist_ok=True)
    sources=collections.defaultdict(list)
    for row in c['images']:
        if file_hash(row['path'])!=row['sha256']: raise ValueError('Content drift')
        sources[row['source']].append(row)
    for source, rows in sources.items():
        if len({r['split'] for r in rows})!=1: raise ValueError('Source split leakage')
    # Keep the existing source split, including held-out files. Never tune on test outputs.
    review=[]
    for source, rows in sorted(sources.items()):
        for i in (0,len(rows)//2,len(rows)-1): review.append(rows[i])
    contact(review,args.output,'content_sources','path')
    targets=[]
    for row in s['records']:
        path=target.parent/row['file']
        if file_hash(path)!=row['sha256']: raise ValueError('Style drift')
        targets.append({**row,'path':str(path)})
    contact(targets,args.output,'game_targets','path')
    manifest={'schema':'coherent-shared-data-v1','status':'REVIEW_REQUIRED',
              'content_parent':str(content),'content_parent_sha256':file_hash(content),
              'style_parent':str(target),'style_parent_sha256':file_hash(target),
              'source_splits':{k:{'split':v[0]['split'],'group':v[0]['group'],
                                  'frames':len(v),'sha256':v[0]['source_sha256']} for k,v in sources.items()},
              'content':c['images'],'style_candidates':targets,'pairs':[],
              'excluded_pairs':[],'extra_holdout_source':'VID_20260925_113231.mp4'}
    if args.pairs:
        cache=args.output/'temporal_pairs'; cache.mkdir(exist_ok=True)
        for source,rows in sorted(sources.items()):
            if rows[0]['split']!='train':continue
            if file_hash(source)!=rows[0]['source_sha256']:raise ValueError('Source video drift')
            wanted={r['frame_index']+3:r for r in rows}
            print('PAIR_SOURCE',Path(source).name,len(rows),flush=True)
            seen=set()
            try:
                for index,pts,pixels in decode_frames(source,indices=sorted(wanted),hlg_exposure_stops=-1.):
                    seen.add(index)
                    row=wanted[index]
                    a=np.array(Image.open(row['path']).convert('RGB'))
                    b=np.array(Image.fromarray(pixels).resize((1280,720),Image.Resampling.BICUBIC))
                    stem=f"{row['source_sha256'][:16]}_{row['frame_index']:08d}"
                    path=cache/f'{stem}_next.png'; Image.fromarray(b).save(path)
                    flow,mask=pair_flow(a,b)
                    flow_path=cache/f'{stem}_flow.npz'
                    np.savez_compressed(flow_path,backward=flow,mask=mask)
                    manifest['pairs'].append({'source':source,'source_sha256':row['source_sha256'],
                        'split':'train','current':row['path'],'current_sha256':row['sha256'],
                        'next':str(path),'next_sha256':file_hash(path),
                        'flow':str(flow_path),'flow_sha256':file_hash(flow_path),
                        'current_index':row['frame_index'],'next_index':index,'next_pts':pts,
                        'valid_fraction':float(mask.mean())})
            except RuntimeError as error:
                if 'Incomplete video decode' not in str(error) or 'exit 0' not in str(error):
                    raise
                missing=set(wanted)-seen
                if not seen or len(missing)>1 or min(missing)<max(seen):raise
                manifest['excluded_pairs'].append({'source':source,'indices':sorted(missing),
                    'reason':'Requested adjacent frame beyond decoded video end; no duplication'})
            write_json(args.output/'manifest.partial.json',manifest)
        missing=sum(len(r['indices']) for r in manifest['excluded_pairs'])
        if len(manifest['pairs'])+missing!=sum(r['split']=='train' for r in c['images']):
            raise ValueError('Missing training pairs')
    write_json(args.output/'manifest.draft.json',manifest)
    print(json.dumps({'sources':len(sources),'content':len(c['images']),
                      'style':len(targets),'pairs':len(manifest['pairs'])}))


if __name__=='__main__':main()
