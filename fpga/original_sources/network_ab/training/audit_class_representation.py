"""Check whether learned regions differ from nearest-palette color reduction."""
import json
import cv2
import numpy as np
import torch
from .train_region_classes import ROOT,RUN,source_manifest,read_image,labels
from .region_mechanism_probe import tensor
from .runtime import write_json


def exact_labels(image,palette):
    encode=lambda x:(x[...,0].astype(np.int32)<<16)+(x[...,1].astype(np.int32)<<8)+x[...,2].astype(np.int32)
    p=encode(palette);order=np.argsort(p);v=encode(image);idx=np.searchsorted(p[order],v)
    assert np.all(idx<len(p)) and np.array_equal(p[order[idx]],v)
    return order[idx]


def components(index):
    sizes=[]
    for c in range(32):
        n,_,stats,_=cv2.connectedComponentsWithStats((index==c).astype(np.uint8),8)
        sizes.extend(stats[1:,cv2.CC_STAT_AREA].tolist())
    a=np.array(sizes)
    return dict(components=len(a),small_component_pixels=int(a[a<=8].sum()),
                small_component_fraction=float(a[a<=8].sum()/index.size),median_area=float(np.median(a)))


def main():
    cv2.setNumThreads(1);run=ROOT/'runs'/RUN;out=run/'classes32';data=json.loads((out/'data.json').read_text(encoding='utf-8'))
    palette=np.array(data['palette'],np.uint8);pal=torch.tensor(palette,dtype=torch.float32,device='cuda')/255;records=[]
    for r in source_manifest()['validation']:
        a=read_image(str(run/'algorithm'/(r['id']+'.png')))
        with torch.no_grad():target=labels(tensor(a),pal)[0].cpu().numpy()
        scores={}
        for name,path in [('palette_only',run/'palette_only_control'),('network',out/'validation_002000')]:
            im=read_image(str(path/(r['id']+'.png')));index=exact_labels(im,palette)
            scores[name]=dict(label_agreement=float((index==target).mean()),rgb_mae_to_algorithm=float(np.abs(im.astype(float)-a).mean()),regions=components(index))
        scores['algorithm_in_same_palette']=dict(regions=components(target))
        records.append(dict(id=r['id'],scores=scores));print('CLASS_REGIONS',r['id'],flush=True)
    write_json(run/'representation_audit.json',dict(records=records,
        limitation='Measures against new L0 region targets, NOT a claim to reproduce AnimeGAN or match user preference. Small-component reduction alone is NOT acceptance.',
        no_extra_training=True))


if __name__=='__main__':main()
