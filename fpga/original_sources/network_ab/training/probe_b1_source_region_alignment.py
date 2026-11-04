"""Offline target audit: source-defined geometry with existing reference colour.

This is neither a network output nor an inference filter. Existing data/labels
remain immutable. It tests whether stylized target geometry itself is a cause
of difficult contour correspondence for the small network.
"""
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .minimum_area_blocks import make_target, initial_regions, components, boundary_mask
from .runtime import configure_workspace, write_json, file_hash


def main():
    configure_workspace(); cv2.setNumThreads(2); torch.set_num_threads(2)
    dest = m.OUT/'source-region-alignment-probe'; dest.mkdir(exist_ok=False)
    checkpoint = m.OUT/'actual-native-coverage-qat/candidate_001000.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    palette = (np.array(state['palette'],np.float32)*128+128).round().clip(0,255)
    rows = m.load_data()['validation']; images = {}; records = []
    for row in rows:
        source = m.t.r.read(row['path']); reference = m.t.r.read(row['parent_target'])
        initial = initial_regions(source)
        images[row['id']] = [Image.fromarray(source),Image.fromarray(m.t.r.read(row['target']))]
        for minimum in (256,512):
            _,labels,stats = make_target(source,minimum,initial)
            size = np.bincount(labels.ravel())
            means = np.stack([np.bincount(labels.ravel(),weights=reference[...,c].ravel())/size for c in range(3)],1)
            choices = ((means[:,None]-palette[None])**2).sum(-1).argmin(1)
            codes = choices[labels].astype(np.uint8); rgb = palette[codes].astype(np.uint8)
            actual_area = np.bincount(components(codes).ravel())
            assert actual_area.min() >= minimum
            path = dest/f'{row["id"]}_{minimum}_offline_target.png'
            Image.fromarray(rgb).save(path); images[row['id']].append(Image.fromarray(rgb))
            records.append(dict(id=row['id'],minimum=minimum,regions=len(actual_area),
                minimum_actual_colour_region=int(actual_area.min()),boundary_density=float(boundary_mask(codes).mean()),
                reference_rgb_mae=float(np.abs(rgb.astype(float)-reference).mean()),
                source_sha256=row['sha256'],reference_sha256=file_hash(row['parent_target']),
                target_sha256=file_hash(path),source_geometry_stats=stats))
        print('SOURCE_REGION_TARGET_AUDIT',row['id'],flush=True)
    board = Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a'); draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('原图','当前离线目标','原图分区·≥256·离线','原图分区·≥512·离线'))):
            board.paste(im.crop(box),(420*j,314*i+34)); draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png'); shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,source_checkpoint_sha256=file_hash(checkpoint),
        current_targets_modified=False,source_prediction=False,network_trained=False,
        visual_acceptance=False,purpose='offline target geometry correspondence diagnostic only'))


if __name__ == '__main__': main()
