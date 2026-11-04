"""Audit where structure and colour disappear, before proposing another fit.

No source/target files are changed. Adjacency statistics describe all labelled
regions, not semantic assertions about glass, walls, or vegetation.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import bits
from .probe_b1_decoder_coverage import fragment
from .fit_b1_sixteen_regions import region_class_map, palette_key
from .runtime import configure_workspace, contract_guard, write_json, file_hash


def region_means(rgb, labels):
    n=int(labels.max())+1
    count=np.bincount(labels.ravel(),minlength=n).clip(1)
    means=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel(),minlength=n)/count for c in range(3)],1)
    return means,count


def lab(rgb):
    return cv2.cvtColor(np.asarray(rgb,np.float32).reshape(-1,1,3)/255,cv2.COLOR_RGB2LAB).reshape(np.asarray(rgb).shape)


def adjacency(labels):
    n=int(labels.max())+1;keys=[]
    for a,b in ((labels[1:],labels[:-1]),(labels[:,1:],labels[:,:-1])):
        keep=a!=b
        keys.append(np.minimum(a[keep],b[keep]).astype(np.int64)*n+np.maximum(a[keep],b[keep]))
    key,count=np.unique(np.concatenate(keys),return_counts=True)
    return key//n,key%n,count


def compare(a,b):
    aa,bb=lab(a),lab(b);diff=bb-aa
    chroma=np.linalg.norm(aa[...,1:],axis=-1)
    return dict(rgb_mae=float(np.abs(b.astype(float)-a).mean()),
        lab_delta_mean=float(np.linalg.norm(diff,axis=-1).mean()),
        lab_signed_bias=diff.mean(axis=(0,1)).tolist(),
        chroma_magnitude_mae=float(np.abs(np.linalg.norm(bb[...,1:],axis=-1)-chroma).mean()))


def edge_comparison(a,b,distance=4):
    # Fixed native720 offset. This is a structural diagnostic, not a pass score.
    wanted=[];actual=[]
    for axis in (0,1):
        x=np.take(a,range(distance,a.shape[axis]),axis=axis).astype(float)-np.take(a,range(a.shape[axis]-distance),axis=axis)
        y=np.take(b,range(distance,b.shape[axis]),axis=axis).astype(float)-np.take(b,range(b.shape[axis]-distance),axis=axis)
        wanted.append(x.reshape(-1,3));actual.append(y.reshape(-1,3))
    x,y=np.concatenate(wanted),np.concatenate(actual)
    mag=np.linalg.norm(x,axis=1);new=np.linalg.norm(y,axis=1)
    strong=mag>=24
    return dict(reference_pairs=int(strong.sum()),
        disappeared_fraction=float((new[strong]<8).mean()),
        reversed_fraction=float(((x[strong]*y[strong]).sum(1)<0).mean()),
        false_edge_fraction=float(((new>=24)&(mag<8)).mean()))


def main():
    configure_workspace();contract_guard();torch.set_num_threads(2);cv2.setNumThreads(2)
    dest=m.OUT/'complete-chain-audit';dest.mkdir(exist_ok=False)
    checkpoint=m.OUT/'aligned-decoder-continued/candidate_001000.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    z=torch.zeros(200,16,1,1);z[:,:15,0,0]=bits('cpu')*127
    with torch.no_grad():palette=(fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]+128).numpy()
    pk=tuple(map(tuple,palette.tolist()));data=m.load_data();records=[];all_images={}
    for row in data['validation']:
        source=m.t.r.read(row['path']);parent=m.t.r.read(row['parent_target']);area=m.t.r.read(row['target'])
        labels=m.load_region(row['region']).astype(np.int64);means,count=region_means(area,labels)
        codes=region_class_map(row['target'],row['region'],pk)
        quant=palette[codes].astype(np.uint8)
        output=m.t.r.read(m.OUT/'aligned-decoder-continued/preview_001000'/(row['id']+'_actual_B1.png'))
        a,b,length=adjacency(labels);d=np.linalg.norm(lab(means)[a]-lab(means)[b],axis=1)
        assigned=((means[:,None]-palette[None])**2).sum(2).argmin(1)
        strong=d>=5;lost=(assigned[a]==assigned[b])
        rec=dict(id=row['id'],regions=len(count),adjacent_region_pairs=len(a),
            distinguishable_pairs=int(strong.sum()),
            palette_collapsed_pairs=int((strong&lost).sum()),
            palette_collapsed_boundary_fraction=float(length[strong&lost].sum()/max(1,length[strong].sum())),
            parent_to_area=compare(parent,area),area_to_palette=compare(area,quant),palette_to_network=compare(quant,output),
            parent_to_area_edges=edge_comparison(parent,area),area_to_palette_edges=edge_comparison(area,quant),
            palette_to_network_edges=edge_comparison(quant,output))
        records.append(rec);print('TARGET_CHAIN',json.dumps(rec),flush=True)
        all_images[row['id']]=list(map(Image.fromarray,(source,parent,area,quant,output)))
    board=Image.new('RGB',(420*5,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(all_images[key],('原图','原风格配方目标','最小面积目标','固定色表后目标','实际B1整数输出'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'audit.json',dict(records=records,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        palette=palette.tolist(),threshold_note='Lab76 distance5 / RGB vector24 are diagnostic cutoffs, not semantic truth',
        no_targets_modified=True,visual_acceptance=False))
    print('TARGET_CHAIN_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
