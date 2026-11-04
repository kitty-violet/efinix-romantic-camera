"""Retain the previous one-pixel boundary coverage in revised supervision.

The hard-colour variant increased target temporal differences. A matched
comparison restoring the original boundary-only3x3 sigma.65 AA reduced them;
region geometry and interior RGB are identical. No deployed postprocessing.
"""
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from . import train_b1_minimum_area as m
from .minimum_area_blocks import boundary_mask
from .runtime import configure_workspace,write_json,file_hash


def main():
    configure_workspace();cv2.setNumThreads(2)
    source=m.OUT/'hierarchical-targets-v1/data.json';data=json.loads(source.read_text())
    dest=m.OUT/'hierarchical-targets-v2';dest.mkdir(exist_ok=False);mapping={};records=[]
    for key in ('training','validation','reserved_test'):
        rows=[]
        for row in data[key]:
            rgb=m.t.r.read(row['target']).astype(np.float32);labels=m.load_region(row['region'])
            edge=cv2.dilate(boundary_mask(labels),np.ones((3,3),np.uint8))
            output=np.where(edge[...,None]>0,cv2.GaussianBlur(rgb,(3,3),.65),rgb).round().clip(0,255).astype(np.uint8)
            assert np.array_equal(output[edge==0],rgb.astype(np.uint8)[edge==0])
            path=dest/(row['sha256'][:20]+'.png');Image.fromarray(output).save(path)
            r=dict(row,target=str(path),target_sha256=file_hash(path),hard_region_target=row['target'])
            rows.append(r);mapping[r['path']]=r
            records.append(dict(id=r['id'],interior_changed_pixels=0,boundary_coverage_pixels=int((edge>0).sum())))
        data[key]=rows
    for pair in data['pairs']:
        for side in ('current','next'):
            row=mapping[pair[side]];pair[side+'_target']=row['target'];pair[side+'_target_sha256']=row['target_sha256']
    data.update(target='Hierarchical region colour; original one-pixel boundary coverage; no fixed colour lattice',
        parent_data_sha256=file_hash(source),boundary_coverage='3x3 sigma.65 in original boundary band only',
        boundary_code_sha256=file_hash(__file__))
    for name in ('add_b1_region_boundary_coverage.py','refine_b1_hierarchical_targets.py','minimum_area_blocks.py'):
        shutil.copy2(Path(__file__).with_name(name),dest/name)
    write_json(dest/'data.json',data);write_json(dest/'coverage-audit.json',records)
    print('BOUNDARY_COVERAGE_TARGETS_COMPLETE',len(records),file_hash(dest/'data.json'),flush=True)


if __name__=='__main__':main()
