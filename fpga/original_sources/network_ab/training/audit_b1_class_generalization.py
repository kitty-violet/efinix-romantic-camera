"""CPU audit: separate TRAIN fit, seen-validation error and actual fragments.

Deterministic source-stratified samples; this tool never optimizes weights.
All predictions come from reloaded bytes, not float previews or corrected masks.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .train_b1_count_regions import count_targets
from .minimum_area_blocks import components,boundary_mask
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    blob=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu')
    data=m.load_data();pk=palette_key(torch.tensor(state['palette']));groups={}
    for row in data['training']:groups.setdefault(row['video'],[]).append(row)
    chosen=[]
    for video,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda x:x['id'])
        chosen.extend(rows[i] for i in sorted(set((0,len(rows)//2,len(rows)-1))))
    rows=chosen+data['validation'];overrides={}
    manifest=state['args'].get('class_manifest','')
    if manifest:
        assert file_hash(manifest)==state['temporal_class_manifest_sha256']
        overrides=json.loads(Path(manifest).read_text(encoding='utf-8'))['overrides']
        assert set(overrides)<={r['id'] for r in data['training']}
    records=[]
    for row in rows:
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
        nodes,_=actual(x,True);bits=nodes['up2_low_dw'][0,:15].numpy()>0
        predicted=bits[:7].sum(0)*25+bits[7:11].sum(0)*5+bits[11:].sum(0)
        if row['id'] in overrides:
            entry=overrides[row['id']];assert file_hash(entry['path'])==entry['sha256']
            with np.load(entry['path']) as z:codes=z['classes'].copy()
        else:codes=region_class_map(row['target'],row['region'],pk)
        truth=count_targets(torch.tensor(codes.astype(np.int64))[None])
        pure=np.stack([(v.amax(1)[0].numpy()>.999) for v in truth]).all(0)
        wanted=sum(v.argmax(1)[0].numpy()*scale for v,scale in zip(truth,(25,5,1)))
        edge=boundary_mask(wanted);interior=pure&(cv2.dilate(edge,np.ones((7,7),np.uint8))==0)
        wrong=predicted!=wanted;reg=components(predicted);area=np.bincount(reg.ravel())
        rec=dict(id=row['id'],video=row['video'],split=row['split'],pure_fraction=float(pure.mean()),
            colour_accuracy_pure=float(1-wrong[pure].mean()),interior_pixels=int(interior.sum()),
            colour_accuracy_interior=float(1-wrong[interior].mean()) if interior.any() else None,
            actual_small_pixel_fraction=float(area[area<64].sum()/area.sum()),
            actual_regions=len(area),target_colours=int(np.unique(codes).size))
        records.append(rec);print('CLASS_FIT_AUDIT',rec,flush=True)
        del nodes
    by_split={}
    for split in sorted({r['split'] for r in records}):
        selected=[r for r in records if r['split']==split]
        by_split[split]={k:float(np.mean([r[k] for r in selected])) for k in
            ('colour_accuracy_pure','colour_accuracy_interior','actual_small_pixel_fraction')}
        by_split[split]['frames']=len(selected)
    write_json(a.out,dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,by_split=by_split,
        selection='first/middle/last frame of each training source; all six seen-validation frames',
        actual_reloaded_integer_bytes=True,optimized_weights=False,blind_test=False,visual_acceptance=False))
    print('CLASS_FIT_SUMMARY',by_split,flush=True)


if __name__=='__main__':main()
