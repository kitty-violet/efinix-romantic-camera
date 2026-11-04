"""Select actual training-only texture patches wholly inside target regions."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from .minimum_area_blocks import OUT,boundary_mask
from .runtime import file_hash,write_json


def main():
    cv2.setNumThreads(2);d=json.loads((OUT/'data.json').read_text(encoding='utf-8'));pool=[]
    for row in d['training'][::4]:
        assert row['split']=='train'
        with np.load(row['region']) as z:lab=z['labels'].astype(np.int32)
        rgb=np.array(Image.open(row['path']).convert('RGB'))
        edge=boundary_mask(lab);edge[0]=1;edge[-1]=1;edge[:,0]=1;edge[:,-1]=1
        distance=cv2.distanceTransform(1-edge,cv2.DIST_L2,5)
        count=np.bincount(lab.ravel());eligible=(distance>24)&(count[lab]>=4096)
        yy,xx=np.where(eligible);choices=[]
        for y,x in zip(yy[::64],xx[::64]):
            patch=rgb[y-16:y+16,x-16:x+16];labels=lab[y-16:y+16,x-16:x+16]
            if patch.shape!=(32,32,3) or len(np.unique(labels))!=1:continue
            detail=patch.astype(np.float32)-cv2.GaussianBlur(patch.astype(np.float32),(9,9),1.7)
            std=float(detail.std());colour_std=float(patch.astype(np.float32).std((0,1)).mean())
            if 2<std<32 and colour_std<55:choices.append((std,int(x),int(y)))
        choices.sort(reverse=True);used=[]
        for std,x,y in choices:
            if any((x-a)**2+(y-b)**2<64**2 for a,b in used):continue
            used.append((x,y));pool.append(dict(id=row['id'],path=row['path'],sha256=row['sha256'],
                region=row['region'],region_sha256=row['region_sha256'],box=[x-16,y-16,x+16,y+16],
                region_id=int(lab[y,x]),region_area=int(count[lab[y,x]]),detail_std=std))
            if len(used)==3:break
    assert len(pool)>=20
    dest=OUT/'interior-texture-pool.json';assert not dest.exists()
    write_json(dest,dict(patches=pool,data_sha256=file_hash(OUT/'data.json'),code_sha256=file_hash(__file__),
        train_only=True,region_boundary_excluded=True,structural_semantics_not_guaranteed=True))
    print('INTERIOR_TEXTURE_POOL',len(pool),'from',len({r['id'] for r in pool}),'train frames',flush=True)


if __name__=='__main__':main()
