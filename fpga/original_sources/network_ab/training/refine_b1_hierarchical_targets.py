"""Recover substantial contrast layers without erasing the parent partitions.

One deterministic binary split is allowed inside each existing >=256-pixel
region. It must explain a bimodal colour variation, and every surviving child
must retain >=256 native720 pixels. No fixed output palette is imposed. This
constructs offline supervision only, never a deployable image filter.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .audit_b1_target_chain import region_means, compare, edge_comparison
from .minimum_area_blocks import boundary_mask, boundary_mode
from .runtime import configure_workspace, write_json, file_hash


def subregions(mask, side, rgb, minimum):
    """Merge only inside one parent: no crossing of established boundaries."""
    z=np.full(mask.shape,-1,np.int32);offset=0
    for k in (False,True):
        present=mask&(side==k)
        n,cc=cv2.connectedComponents(present.astype(np.uint8),connectivity=4)
        z[present]=cc[present]-1+offset;offset+=n-1
    if offset<2:return np.zeros(int(mask.sum()),np.int32)
    ids=z[mask];size=np.bincount(ids,minlength=offset).astype(float)
    sums=np.stack([np.bincount(ids,weights=rgb[...,c][mask],minlength=offset) for c in range(3)],1)
    adj=[set() for _ in range(offset)];parent=np.arange(offset)
    for a,b in ((z[1:],z[:-1]),(z[:,1:],z[:,:-1])):
        valid=(a>=0)&(b>=0)&(a!=b)
        pairs=np.unique(np.sort(np.stack((a[valid],b[valid]),1),axis=1),axis=0)
        for p,q in pairs:adj[p].add(int(q));adj[q].add(int(p))
    while True:
        eligible=np.flatnonzero((size>0)&(size<minimum))
        if not len(eligible):break
        a=int(eligible[np.argmin(size[eligible])])
        if not adj[a]:assert size[a]>=minimum;break
        b=min(adj[a],key=lambda j:float(np.sum((sums[a]/size[a]-sums[j]/size[j])**2)))
        parent[a]=b;size[b]+=size[a];sums[b]+=sums[a];size[a]=0
        adj[b].discard(a)
        for j in adj[a]:
            adj[j].discard(a)
            if j!=b:adj[j].add(b);adj[b].add(j)
        adj[a].clear()
    for a in range(offset):
        while parent[a]!=parent[parent[a]]:parent[a]=parent[parent[a]]
    return np.unique(parent[ids],return_inverse=True)[1].astype(np.int32)


def refine(rgb, labels, minimum=256):
    assert np.bincount(labels.ravel()).min()>=minimum
    # Guidance alone is slightly smoothed; region colours use original RGB.
    guide=cv2.GaussianBlur(rgb.astype(np.float32)/255,(5,5),1.)
    guide=cv2.cvtColor(guide,cv2.COLOR_RGB2LAB)
    count=np.bincount(labels.ravel());n=len(count);out=labels.astype(np.int32).copy()
    next_id=n;splits=0;rejected=0
    # Bounding boxes avoid rescanning the whole image for each candidate.
    yy,xx=np.indices(labels.shape);xmin=np.full(n,labels.shape[1]);ymin=np.full(n,labels.shape[0]);xmax=np.zeros(n,int);ymax=np.zeros(n,int)
    np.minimum.at(xmin,labels.ravel(),xx.ravel());np.maximum.at(xmax,labels.ravel(),xx.ravel())
    np.minimum.at(ymin,labels.ravel(),yy.ravel());np.maximum.at(ymax,labels.ravel(),yy.ravel())
    for key in np.flatnonzero(count>=2*minimum):
        ys=slice(ymin[key],ymax[key]+1);xs=slice(xmin[key],xmax[key]+1)
        mask=labels[ys,xs]==key;v=guide[ys,xs][mask]
        axis=int(v.var(0).argmax());lo,hi=np.quantile(v[:,axis],[.2,.8])
        if hi-lo<5:continue
        low=v[v[:,axis]<=lo].mean(0);high=v[v[:,axis]>=hi].mean(0)
        initial=float(((v-v.mean(0))**2).sum())
        for _ in range(8):
            side=((v-high)**2).sum(1)<((v-low)**2).sum(1)
            if side.all() or not side.any():break
            low,high=v[~side].mean(0),v[side].mean(0)
        residual=float(((v-np.where(side[:,None],high,low))**2).sum())
        if min(side.sum(),(~side).sum())<minimum or np.linalg.norm(high-low)<7 or residual>.45*initial:
            rejected+=1;continue
        field=np.zeros(mask.shape,np.int32);field[mask]=side.astype(np.int32)+1
        clean=boundary_mode(field,1)
        field=np.where(mask& (clean>0),clean,field)
        sub=subregions(mask,field==2,rgb[ys,xs],minimum)
        k=int(sub.max())+1
        if k<2:continue
        mapping=np.r_[key,np.arange(next_id,next_id+k-1)];out[ys,xs][mask]=mapping[sub]
        next_id+=k-1;splits+=1
    _,out=np.unique(out,return_inverse=True);out=out.reshape(labels.shape).astype(np.int32)
    colours,size=region_means(rgb,out)
    hard=colours[out].round().clip(0,255).astype(np.uint8)
    # Colour changes only at labels; display AA is not applied to training RGB.
    assert size.min()>=minimum
    return hard,out,dict(parent_regions=n,regions=len(size),split_parent_regions=splits,
        rejected_weak_splits=rejected,minimum_area=int(size.min()),fixed_palette=False)


def regression():
    rgb=np.zeros((64,96,3),np.uint8);rgb[:,:48]=[75,95,110];rgb[:,48:]=[135,140,145]
    labels=np.zeros((64,96),np.int32)
    out,lab,stats=refine(rgb,labels)
    assert stats['regions']==2 and np.abs(out.astype(float)-rgb).max()==0
    # Tiny high contrast speck cannot introduce a subminimum colour region.
    sample=rgb.copy();sample[5:8,5:8]=255
    out,lab,stats=refine(sample,labels);assert np.bincount(lab.ravel()).min()>=256
    # A pre-existing two-parent partition cannot be merged by refinement.
    parents=labels.copy();parents[:,48:]=1
    out,lab,stats=refine(np.full_like(rgb,128),parents)
    assert np.all(lab[:,:48]!=lab[:,48:]) and stats['regions']==2
    print('HIERARCHICAL_TARGET_REGRESSION_PASS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    configure_workspace();cv2.setNumThreads(2);regression();d=m.load_data()
    dest=m.OUT/'hierarchical-targets-v1';dest.mkdir(exist_ok=True)
    rows=d['training']+d['validation']+d['reserved_test'] if a.prepare else d['validation']
    records=[];pictures={};mapping={}
    for i,row in enumerate(rows):
        teacher=m.t.r.read(row['parent_target']);old=m.t.r.read(row['target']);labels=m.load_region(row['region'])
        rgb,lab,stats=refine(teacher,labels)
        path=dest/(row['sha256'][:20]+'.png');label=dest/(row['sha256'][:20]+'.npz')
        Image.fromarray(rgb).save(path);np.savez_compressed(label,labels=lab.astype(np.uint16))
        rec=dict(id=row['id'],**stats,old_colour=compare(teacher,old),new_colour=compare(teacher,rgb),
            old_edges=edge_comparison(teacher,old),new_edges=edge_comparison(teacher,rgb))
        records.append(rec)
        if row in d['validation']:pictures[row['id']]=list(map(Image.fromarray,(teacher,old,rgb)))
        row=dict(row,previous_target=row['target'],previous_region=row['region'],target=str(path),region=str(label),
            target_sha256=file_hash(path),region_sha256=file_hash(label))
        mapping[row['path']]=row
        print('HIERARCHICAL_TARGET',i+1,len(rows),row['id'],stats,flush=True)
    if a.prepare:
        for key in ('training','validation','reserved_test'):d[key]=[mapping[r['path']] for r in d[key]]
        for pair in d['pairs']:
            for side in ('current','next'):
                r=mapping[pair[side]];pair[side+'_target']=r['target'];pair[side+'_target_sha256']=r['target_sha256']
        d.update(target='Hierarchical region splits; original region RGB; no fixed palette',parent_data_sha256=file_hash(m.OUT/'data.json'),
            label_code_sha256=file_hash(__file__),inference_minimum_area_guaranteed=False)
        write_json(dest/'data.json',d)
    write_json(dest/('full-audit.json' if a.prepare else 'preview-audit.json'),records)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(pictures[key],('原风格配方目标','旧区域目标·量化前','修订区域目标·非网络输出'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')


if __name__=='__main__':main()
