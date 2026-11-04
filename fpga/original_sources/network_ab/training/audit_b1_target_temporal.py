"""Read-only temporal consistency audit of the actual training labels.

Uses existing backward flow and source agreement, excludes transported label
boundaries, and compares parent RGB, area RGB, and final palette assignments.
This diagnoses target contradictions; it does not prove perfect optical flow.
"""
import argparse
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .minimum_area_blocks import boundary_mask
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .train_b1_distillation import read_flow
from .runtime import file_hash,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--class-manifest',default='')
    a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    palette=torch.tensor(s['palette']);pk=palette_key(palette);pal=np.array(pk,np.float32)
    data=m.load_data();rows={r['path']:r for r in data['training']};records=[]
    override={}
    if a.class_manifest:
        manifest=json.loads(Path(a.class_manifest).read_text(encoding='utf-8'))
        assert manifest['palette_sha256']==hashlib.sha256(pal.tobytes()).hexdigest()
        assert manifest['original_manifest_sha256']==file_hash(m.OUT/'data.json')
        override=manifest['overrides']
    def codes(row):
        if row['id'] not in override:return region_class_map(row['target'],row['region'],pk)
        entry=override[row['id']];assert file_hash(entry['path'])==entry['sha256']
        with np.load(entry['path']) as z:return z['classes'].copy()
    yy,xx=np.mgrid[:720,:1280].astype(np.float32)
    for pair in data['pairs']:
        assert pair['split']=='train'
        left,right=[rows[pair[k]] for k in ('current','next')]
        flow,mask=read_flow(pair['flow']);mx=xx+flow[:,:,0];my=yy+flow[:,:,1]
        warp=lambda z,interp=cv2.INTER_LINEAR:cv2.remap(z,mx,my,interp,borderMode=cv2.BORDER_REPLICATE)
        src0,src1=[m.t.r.read(r['path']).astype(np.float32) for r in (left,right)]
        source_delta=np.abs(warp(src0)-src1).mean(2)
        valid=(mask>.5)&(source_delta<4)&(mx>=0)&(mx<1279)&(my>=0)&(my<719)
        labels0,labels1=[m.load_region(r['region']) for r in (left,right)]
        edges0,edges1=[boundary_mask(z) for z in (labels0,labels1)]
        edges=warp(edges0,cv2.INTER_NEAREST)|edges1
        core=valid&(cv2.dilate(edges,np.ones((9,9),np.uint8))==0)
        record=dict(source=Path(pair['source']).name,current_index=pair['current_index'],
            source_agreement_under4_fraction=float(valid.mean()),core_fraction=float(core.mean()),
            core_pixels=int(core.sum()),source_mae_core=float(source_delta[core].mean()) if core.any() else None)
        for label,key in (('parent','parent_target'),('area','target')):
            images=[m.t.r.read(r[key]).astype(np.float32) for r in (left,right)]
            delta=np.abs(warp(images[0])-images[1]).mean(2)[core]
            record[label]=dict(mae=float(delta.mean()),over8=float((delta>8).mean()),
                p95=float(np.percentile(delta,95))) if len(delta) else None
        codes0,codes1=[codes(r) for r in (left,right)]
        old=warp(codes0,cv2.INTER_NEAREST)
        change=(old!=codes1)[core];delta=np.abs(pal[old]-pal[codes1]).mean(2)[core]
        record['palette']=dict(class_change_fraction=float(change.mean()),mae=float(delta.mean()),
            over8=float((delta>8).mean()),p95=float(np.percentile(delta,95))) if len(delta) else None
        records.append(record)
        print('TARGET_TEMPORAL_PAIR',record['source'],record['current_index'],record['palette'],flush=True)
    counts=np.array([r['core_pixels'] for r in records]);good=counts>0
    def weighted(key,field):return float(np.average([r[key][field] for r in records if r['core_pixels']],weights=counts[good]))
    summary={key:{field:weighted(key,field) for field in ('mae','over8')} for key in ('parent','area','palette')}
    summary['palette']['class_change_fraction']=weighted('palette','class_change_fraction')
    write_json(a.out,dict(records=records,weighted_summary=summary,checkpoint_sha256=file_hash(a.checkpoint),
        manifest_sha256=file_hash(m.OUT/'data.json'),read_only=True,no_training_data_modified=True,
        optional_class_manifest=a.class_manifest,optional_class_manifest_sha256=file_hash(a.class_manifest) if a.class_manifest else None,
        mask='existing flow validity + source warp RGB MAE<4 + >=4px from either target region boundary',
        limitation='flow/occlusion/real exposure changes may remain; no claim every discrepancy is a label error'))
    print('TARGET_TEMPORAL_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
