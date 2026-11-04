"""Read-only audit: tiny RGB regions formed by crossing larger colour fields."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components
from .audit_b1_native_colour_regions import nearest_codes
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--preview',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    palette=(np.array(state['palette'])*128+128).round().clip(0,255)
    records=[]
    for row in m.load_data()['validation']:
        path=Path(a.preview)/(row['id']+'_actual_B1.png');rgb=np.array(Image.open(path).convert('RGB'))
        code,membership=nearest_codes(rgb,palette);assert membership['off_palette_fraction']==0
        lab=components(code);area=np.bincount(lab.ravel());small=area[lab]<256
        axis_big=[];axis_records=[]
        for field in (code//25,(code//5)%5,code%5):
            group=components(field);size=np.bincount(group.ravel());big=size[group]>=256
            axis_big.append(big);axis_records.append(dict(small_pixel_fraction=float((~big).mean()),regions=len(size)))
        crossing=np.logical_and.reduce(axis_big)&small
        record=dict(id=row['id'],rgb_small_pixel_fraction=float(small.mean()),
            all_coordinates_large_but_joint_small_fraction=float(crossing.mean()),
            share_of_small_pixels_from_large_field_intersections=float(crossing.sum()/max(1,small.sum())),
            per_coordinate=axis_records,png_sha256=file_hash(path))
        records.append(record);print('COLOUR_INTERSECTION_AUDIT',record,flush=True)
    write_json(a.out,dict(checkpoint=a.checkpoint,checkpoint_sha256=file_hash(a.checkpoint),
        records=records,minimum720=256,changed_images=False,diagnostic_not_causal_proof=True))


if __name__=='__main__':main()
