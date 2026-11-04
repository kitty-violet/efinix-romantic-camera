"""Training-only connected colour areas and short, simplified boundaries.

No operation in this module belongs to the B1 inference graph. Minimum area is
guaranteed for the generated region labels, NOT claimed for a learned output.
"""
import argparse
import heapq
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .runtime import configure_workspace, file_hash, write_json
from . import train_b1_style_recovery as t

OUT = t.OUT.parent / 'b1-minimum-area-20261006'


def components(codes):
    """Unique ID for every 4-connected component; no diagonal-only joining."""
    out = np.zeros(codes.shape, np.int32)
    start = 0
    for code in np.unique(codes):
        mask = (codes == code).astype(np.uint8)
        n, cc = cv2.connectedComponents(mask, connectivity=4)
        keep = mask != 0
        out[keep] = cc[keep] + start - 1
        start += n - 1
    assert out.min() == 0 and out.max() == start - 1
    return out


def area_merge(labels, rgb, minimum):
    """Merge small connected regions only into touching colour neighbours."""
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    size = np.bincount(labels.ravel(), minlength=n).astype(np.int64)
    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    sums = np.stack([np.bincount(labels.ravel(), weights=lab[..., c].ravel(), minlength=n)
                     for c in range(3)], 1)
    parent = np.arange(n)
    adjacent = [dict() for _ in range(n)]
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
        mask = a != b
        lo, hi = np.minimum(a[mask], b[mask]), np.maximum(a[mask], b[mask])
        pairs.append(lo.astype(np.int64)*n + hi)
    keys, counts = np.unique(np.concatenate(pairs), return_counts=True)
    for key, count in zip(keys, counts):
        a, b = divmod(int(key), n)
        adjacent[a][b] = adjacent[b][a] = int(count)
    queue = [(int(s), i) for i, s in enumerate(size) if 0 < s < minimum]
    heapq.heapify(queue)
    merged = 0
    while queue:
        count, a = heapq.heappop(queue)
        if parent[a] != a or count != size[a] or size[a] >= minimum:
            continue
        if not adjacent[a]:
            assert size[a] == labels.size
            continue
        ca = sums[a]/size[a]
        # A weak preference for a shared boundary avoids needle-like mergers;
        # colour distance dominates. All choices remain spatial neighbours.
        b = min(adjacent[a], key=lambda j: float(np.sum((ca-sums[j]/size[j])**2))
                + 12/np.sqrt(adjacent[a][j]))
        parent[a] = b
        size[b] += size[a]
        sums[b] += sums[a]
        size[a] = 0
        adjacent[b].pop(a, None)
        for j, length in list(adjacent[a].items()):
            if j == b:
                continue
            adjacent[j].pop(a, None)
            adjacent[b][j] = adjacent[b].get(j, 0) + length
            adjacent[j][b] = adjacent[j].get(b, 0) + length
        adjacent[a].clear()
        if size[b] < minimum:
            heapq.heappush(queue, (int(size[b]), b))
        merged += 1
    for i in range(n):
        j = i
        while parent[j] != j:
            j = parent[j]
        parent[i] = j
    _, mapped = np.unique(parent[labels], return_inverse=True)
    result = mapped.reshape(labels.shape).astype(np.int32)
    areas = np.bincount(result.ravel())
    assert areas.min() >= minimum or len(areas) == 1
    return result, merged


def boundary_mode(labels, iterations=2):
    """Categorical 3x3 majority, not numeric median of arbitrary region IDs.

    Ties keep the centre. Two rounds remove 1px teeth without blurring colours.
    Components and area are checked again afterwards because smoothing can split.
    """
    z = labels.copy()
    h, w = z.shape
    for _ in range(iterations):
        pad = np.pad(z, 1, mode='edge')
        candidates = np.stack([pad[y:y+h, x:x+w] for y, x in
                               [(1,1),(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]])
        selected = candidates[0].copy()
        most = (candidates == selected).sum(0)
        for candidate in candidates[1:]:
            count = (candidates == candidate).sum(0)
            replace = count > most
            selected[replace] = candidate[replace]
            most[replace] = count[replace]
        z = selected
    return z


def boundary_mask(labels):
    b = np.zeros(labels.shape, np.uint8)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:] |= labels[1:] != labels[:-1]
    return b


def describe(labels):
    areas = np.bincount(labels.ravel())
    areas = areas[areas > 0]
    return dict(regions=len(areas), min_area=int(areas.min()), median_area=float(np.median(areas)),
                boundary_pixels=int(boundary_mask(labels).sum()))


def initial_regions(rgb):
    small = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_AREA)
    # Mean shift only defines training segmentation; never applied to predictions.
    smooth = cv2.pyrMeanShiftFiltering(small, sp=5, sr=14, maxLevel=1)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.int32)
    codes = (lab[...,0]//8)*1024 + (lab[...,1]//10)*32 + lab[...,2]//10
    return small, components(codes)


def make_target(rgb, minimum720=256, initial=None):
    small, raw = initial_regions(rgb) if initial is None else initial
    minimum = (minimum720 + 3)//4
    merged, count1 = area_merge(raw, small, minimum)
    simple = boundary_mode(merged)
    separated = components(simple)
    final, count2 = area_merge(separated, small, minimum)
    labels = np.repeat(np.repeat(final, 2, 0), 2, 1)
    # Remove the half-resolution 2x2 staircase in label geometry itself.
    # This is a categorical vote, so it never averages unrelated RGB colours.
    labels = boundary_mode(labels, iterations=2)
    labels, count3 = area_merge(components(labels), rgb, minimum720)
    size = np.bincount(labels.ravel())
    colours = np.stack([np.bincount(labels.ravel(), weights=rgb[...,c].ravel())/size for c in range(3)], 1)
    # Constant colours inside connected regions; AA is restricted to a 1px
    # coverage layer at the label boundary. No broad RGB Gaussian smoothing.
    hard = colours[labels].astype(np.float32)
    edge = boundary_mask(labels)
    edge = cv2.dilate(edge, np.ones((3,3),np.uint8))
    aa = cv2.GaussianBlur(hard, (3,3), .65)
    result = np.where(edge[...,None], aa, hard).round().clip(0,255).astype(np.uint8)
    stats = dict(raw360=describe(raw), before_simplify360=describe(merged), final720=describe(labels),
                 minimum720=minimum720, merge_count=count1+count2+count3, aa='boundary-only sigma0.65 3x3')
    assert stats['final720']['min_area'] >= minimum720
    return result, labels.astype(np.uint16), stats


def preview():
    configure_workspace(); cv2.setNumThreads(2)
    dest=OUT/'target-probe';dest.mkdir(parents=True,exist_ok=False)
    d=t.data();images={};records=[]
    for row in d['validation']:
        rgb=t.r.read(row['target']);initial=initial_regions(rgb)
        images[row['id']]=[Image.fromarray(rgb)]
        for area in (144,256,400):
            started=time.perf_counter();out,labels,stats=make_target(rgb,area,initial)
            name=row['id']+'_'+str(area)
            Image.fromarray(out).save(dest/(name+'.png'))
            np.savez_compressed(dest/(name+'_labels.npz'),labels=labels)
            images[row['id']].append(Image.fromarray(out))
            records.append(dict(id=row['id'],area=area,seconds=time.perf_counter()-started,**stats))
            print('AREA_PROBE',row['id'],area,stats['final720'],flush=True)
    board=Image.new('RGB',(420*4,314*len(t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for i,(key,box,label) in enumerate(t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],['原风格目标','训练目标 ≥144像素','训练目标 ≥256像素','训练目标 ≥400像素'])):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,network_output=False,code_sha256=file_hash(__file__)))


def prepare(minimum720):
    configure_workspace();cv2.setNumThreads(2)
    dest=OUT/'labels';dest.mkdir(parents=True,exist_ok=False)
    d=t.data();mapping={};records=[]
    for i,row in enumerate(d['training']+d['validation']+d['reserved_test']):
        started=time.perf_counter()
        out,labels,stats=make_target(t.r.read(row['target']),minimum720)
        path=dest/(row['sha256'][:20]+'.png');label=dest/(row['sha256'][:20]+'.npz')
        Image.fromarray(out).save(path);np.savez_compressed(label,labels=labels)
        row['parent_target']=row['target'];row['parent_target_sha256']=row['target_sha256']
        row.update(target=str(path),target_sha256=file_hash(path),region=str(label),region_sha256=file_hash(label))
        mapping[row['path']]=row
        records.append(dict(id=row['id'],split=row['split'],seconds=time.perf_counter()-started,**stats))
        print('AREA_LABEL',i+1,len(d['training']+d['validation']+d['reserved_test']),stats['final720'],flush=True)
    for pair in d['pairs']:
        for side in ('current','next'):
            row=mapping[pair[side]]
            pair[side+'_target']=row['target'];pair[side+'_target_sha256']=row['target_sha256']
    d.update(target='gentle-v3 connected minimum-area simplified training labels',minimum_area720=minimum720,
             parent_data_sha256=file_hash(t.OUT/'data.json'),label_code_sha256=file_hash(__file__),
             inference_minimum_area_guaranteed=False)
    write_json(OUT/'data.json',d);write_json(OUT/'label-audit.json',records)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--minimum',type=int,default=256)
    args=ap.parse_args()
    if args.prepare:prepare(args.minimum)
    else:preview()
