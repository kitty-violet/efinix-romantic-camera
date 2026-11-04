"""Pairwise colour consensus for reliable TRAIN region correspondences only.

Region geometry and the palette are unchanged. Matching requires mutual80%
overlap, source agreement, at least50% reliable region coverage and parent
area colour difference<=12. Choose one of the two existing palette colours.
Original data/targets are immutable; this is a separate optional label bundle.
"""
import argparse
import hashlib
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .train_b1_distillation import read_flow
from .runtime import file_hash,write_json


def reliable_matches(a,b,valid,colour_a,colour_b):
    na,nb=len(colour_a),len(colour_b)
    area_a=np.bincount(a.ravel(),minlength=na);area_b=np.bincount(b.ravel(),minlength=nb)
    count_a=np.bincount(a[valid],minlength=na);count_b=np.bincount(b[valid],minlength=nb)
    keys,counts=np.unique(a[valid].astype(np.int64)*nb+b[valid],return_counts=True)
    aa,bb=keys//nb,keys%nb
    best_a=np.zeros(na,np.int64);best_b=np.zeros(nb,np.int64)
    np.maximum.at(best_a,aa,counts);np.maximum.at(best_b,bb,counts)
    keep=(counts==best_a[aa])&(counts==best_b[bb])
    keep&=(counts>=.8*count_a[aa])&(counts>=.8*count_b[bb])
    keep&=(count_a[aa]>=.5*area_a[aa])&(count_b[bb]>=.5*area_b[bb])
    keep&=(np.abs(colour_a[aa]-colour_b[bb]).mean(1)<=12)
    return [(int(x),int(y),int(n)) for x,y,n in zip(aa[keep],bb[keep],counts[keep])]


def region_colours(row):
    lab=m.load_region(row['region']).astype(np.int32);rgb=m.t.r.read(row['target']).astype(np.float64)
    size=np.bincount(lab.ravel());assert size.min()>=256
    means=np.stack([np.bincount(lab.ravel(),weights=rgb[:,:,c].ravel())/size for c in range(3)],1)
    return lab,size,means


def regression():
    a=np.zeros((32,64),np.int32);a[:,32:]=1;b=a.copy();mask=np.ones_like(a,bool)
    col=np.array([[20,60,90],[100,150,180]],float)
    pairs=reliable_matches(a,b,mask,col,col+1)
    assert [(x,y) for x,y,_ in pairs]==[(0,0),(1,1)]
    assert not reliable_matches(a,b,mask,col,col+20)
    low=mask.copy();low[12:]=False;assert not reliable_matches(a,b,low,col,col)
    # A split with50/50 overlap is not a mutual80% region correspondence.
    b[::2]=1-b[::2];assert not reliable_matches(a,b,mask,col,col)
    print('TEMPORAL_REGION_MATCH_REGRESSION PASS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();regression();torch.set_num_threads(2);cv2.setNumThreads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);palette=torch.tensor(state['palette'])
    pk=palette_key(palette);pal=np.array(pk,np.float32);data=m.load_data();rows={r['path']:r for r in data['training']}
    dest=Path(a.out);dest.mkdir(exist_ok=False);yy,xx=np.mgrid[:720,:1280].astype(np.float32)
    overrides={};records=[]
    for pair in data['pairs']:
        assert pair['split']=='train';left,right=[rows[pair[k]] for k in ('current','next')]
        assert left['id'] not in overrides and right['id'] not in overrides,'Pair graph now needs multi-frame reconciliation'
        l0,size0,mean0=region_colours(left);l1,size1,mean1=region_colours(right)
        code0=region_class_map(left['target'],left['region'],pk).copy();code1=region_class_map(right['target'],right['region'],pk).copy()
        c0=((mean0[:,None]-pal[None])**2).sum(2).argmin(1);c1=((mean1[:,None]-pal[None])**2).sum(2).argmin(1)
        assert np.array_equal(c0[l0],code0) and np.array_equal(c1[l1],code1)
        flow,mask=read_flow(pair['flow']);mx=xx+flow[:,:,0];my=yy+flow[:,:,1]
        warp=lambda z,mode=cv2.INTER_LINEAR:cv2.remap(z,mx,my,mode,borderMode=cv2.BORDER_REPLICATE)
        src0,src1=[m.t.r.read(r['path']).astype(np.float32) for r in (left,right)]
        valid=(mask>.5)&(np.abs(warp(src0)-src1).mean(2)<4)&(mx>=0)&(mx<1279)&(my>=0)&(my<719)
        correspondences=reliable_matches(warp(l0,cv2.INTER_NEAREST),l1,valid,mean0,mean1)
        corrected=0
        for u,v,count in correspondences:
            if c0[u]==c1[v]:continue
            choices=np.array([c0[u],c1[v]])
            costs=size0[u]*((pal[choices]-mean0[u])**2).sum(1)+size1[v]*((pal[choices]-mean1[v])**2).sum(1)
            selected=choices[costs.argmin()];c0[u]=c1[v]=selected;corrected+=1
        adjusted=[c0[l0].astype(np.uint8),c1[l1].astype(np.uint8)]
        changed=[]
        for row,codes,old in zip((left,right),adjusted,(code0,code1)):
            path=dest/(row['id']+'.npz');np.savez_compressed(path,classes=codes)
            fraction=float((codes!=old).mean());changed.append(fraction)
            overrides[row['id']]=dict(path=str(path),sha256=file_hash(path),split='train',
                source_sha256=row['sha256'],region_sha256=row['region_sha256'],changed_fraction=fraction)
        records.append(dict(source=pair['source'],current_index=pair['current_index'],
            eligible_pairs=len(correspondences),colour_disagreements_corrected=corrected,changed_fractions=changed))
        print('TEMPORAL_CLASS_PREPARE',Path(pair['source']).name,pair['current_index'],corrected,changed,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'manifest.json',dict(schema='b1-temporal-class-consensus-v1',overrides=overrides,records=records,
        palette_sha256=hashlib.sha256(pal.tobytes()).hexdigest(),palette_rgb=pal.tolist(),
        original_manifest_sha256=file_hash(m.OUT/'data.json'),code_sha256=file_hash(__file__),
        minimum_native_region_area=256,region_geometry_unchanged=True,validation_unchanged=True,
        inference_postprocessing=False,temporal_inference_added=False,visual_acceptance=False))


if __name__=='__main__':main()
