"""CPU diagnostics of narrow colour bands and misplaced actual B1 boundaries.

Width alone does not identify a defect: thin real objects may be legitimate.
Report it beside the labelled target and colour error, never as a pass gate.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components, boundary_mask
from .fit_b1_sixteen_regions import palette_key, region_class_map
from .runtime import write_json, file_hash


def geometry(codes):
    regions=components(codes);area=np.bincount(regions.ravel())
    radius=np.zeros(len(area),np.float32)
    # Padding makes distance to an image border finite and conservative.
    distance=np.zeros(codes.shape,np.float32)
    for code in np.unique(codes):
        mask=(codes==code).astype(np.uint8)
        d=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,5)[1:-1,1:-1]
        distance[mask!=0]=d[mask!=0]
    np.maximum.at(radius,regions.ravel(),distance.ravel())
    large=area>=64
    stats=dict(regions=len(area),small_pixel_fraction=float(area[~large].sum()/area.sum()))
    for width in (2,3,4):
        thin=large&(radius<width)
        stats[f'large_but_radius_below{width}_fraction']=float(area[thin].sum()/area.sum())
        stats[f'large_but_radius_below{width}_count']=int(thin.sum())
    return stats,regions,radius


def boundary_distance(a,b):
    ea,eb=boundary_mask(a),boundary_mask(b)
    db=cv2.distanceTransform(1-eb,cv2.DIST_L2,5)*2
    values=db[ea!=0]
    return dict(mean720=float(values.mean()) if len(values) else None,
        over4px=float((values>4).mean()) if len(values) else None,
        count=int(len(values)))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding']
    net.quantized=True;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
    palette=torch.tensor(state['palette']);pk=palette_key(palette);records=[]
    for row in m.load_data()['validation']:
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
        nodes,_=actual(x,True);bits=nodes['up2_low_dw'][0,:15].numpy()>0
        codes=bits[:7].sum(0)*25+bits[7:11].sum(0)*5+bits[11:].sum(0)
        target=region_class_map(row['target'],row['region'],pk).astype(np.int64)[::2,::2]
        measured,_,_=geometry(codes);reference,_,_=geometry(target)
        record=dict(id=row['id'],actual=measured,target=reference,
            actual_to_target_boundary=boundary_distance(codes,target),
            target_to_actual_boundary=boundary_distance(target,codes))
        records.append(record);print('BOUNDARY_BANDS',record,flush=True)
    write_json(a.out,dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,
        diagnostic_only=True,width_is_not_semantic_validity=True,decision_grid='360p',radius_unit='360p pixels'))


if __name__=='__main__':main()
