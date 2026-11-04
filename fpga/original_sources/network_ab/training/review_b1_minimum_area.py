"""Review reloaded B1 bytes, with no area algorithm in the inference path."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .publish_b1_reconstruction_research import ActualBytes
from .audit_b1_reconstruction_research import phase_tests
from .runtime import configure_workspace, file_hash, write_json
from .engine import set_reproducible


def region_diagnostics(rgb, target, labels):
    """Independent RGB measurements; no filtering of the actual output."""
    labels=labels.astype(np.int32);n=int(labels.max())+1
    count=np.bincount(labels.ravel(),minlength=n).clip(1)
    means=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel(),minlength=n)/count for c in range(3)],1)
    residual=np.abs(rgb.astype(np.float32)-means[labels])
    edge=np.zeros(labels.shape,np.uint8)
    edge[:,1:]|=labels[:,1:]!=labels[:,:-1];edge[1:]|=labels[1:]!=labels[:-1]
    interior=cv2.dilate(edge,np.ones((7,7),np.uint8))==0
    gp=[];gt=[]
    for axis in (0,1):
        p=np.abs(np.diff(rgb.astype(np.float32),axis=axis)).mean(2)
        t=np.abs(np.diff(target.astype(np.float32),axis=axis)).mean(2)
        mask=(interior[1:]&interior[:-1]) if axis==0 else (interior[:,1:]&interior[:,:-1])
        gp.append(p[mask]);gt.append(t[mask])
    gp=np.concatenate(gp);gt=np.concatenate(gt)
    return dict(region_rgb_deviation=float(residual[interior].mean()),
        interior_transition_over6_fraction=float((gp>6).mean()),
        interior_gradient_rgb=float(gp.mean()),target_interior_gradient_rgb=float(gt.mean()),
        interior_pixels=int(interior.sum()),
        minimum_area_claim='256px applies to training labels only; no hard guarantee for learned inference')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--reserved-test',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(2026100641);cv2.setNumThreads(2);d=m.load_data()
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    q=m.t.r.initialize();state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    q.load_state_dict(state['model']);q.quantized=True
    if 'residual_encoding' in state:q.package['residual_encoding']=state['residual_encoding']
    export(q,d['validation'],dest/'package');shutil.copy2(a.checkpoint,dest/'package/weights.pt')
    parent=torch.load(m.t.OUT/'clear-regions/best.pt',map_location='cpu',weights_only=True)
    old=m.t.r.initialize();old.load_state_dict(parent['model']);old.quantized=True
    # The primary review model always reloads the actual 20480 bytes.
    actual=ActualBytes(dest/'package/coefficients-padded.bin');images={};records=[]
    for row in d['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        source=Image.fromarray(m.t.r.read(row['path']));teacher=Image.fromarray(m.t.r.read(row['parent_target']))
        target=Image.fromarray(m.t.r.read(row['target']));old_image=m.t.r.c.rgb_image(old(x)[0][0])
        output=m.t.r.c.rgb_image(actual(x)[0][0]);output.save(dest/(row['id']+'_integer720.png'))
        images[row['id']]=[source,teacher,target,old_image,output]
        labels=m.load_region(row['region'])
        records.append(dict(id=row['id'],parent=region_diagnostics(np.array(old_image),np.array(target),labels),
            actual=region_diagnostics(np.array(output),np.array(target),labels)))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for name,columns in [('comparison',[1,3,4]),('target-check',[1,2,4]),('source-check',[0,2,4])]:
        board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
        titles=['原图','原认可画风目标','最小面积训练目标','修正前实际B1','本轮真实20KiB包']
        for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
            for j,column in enumerate(columns):
                board.paste(images[key][column].crop(box),(j*420,i*314+34))
                draw.text((j*420+4,i*314+5),label+' | '+titles[column],font=font,fill='white')
        board.save(dest/(name+'.png'))
    results=dict(checkpoint=a.checkpoint,checkpoint_sha256=file_hash(a.checkpoint),selected_step=state.get('step'),
        source_data_sha256=file_hash(m.OUT/'data.json'),package_sha256=file_hash(dest/'package/coefficients-padded.bin'),
        region_diagnostics=records,validation=m.evaluate(actual,d['validation']),phase=phase_tests(actual,d['validation']),
        inference_postfilter=False,actual_board=False,visual_acceptance=False)
    if a.reserved_test:
        results['reserved_test']=m.evaluate(actual,d['reserved_test'])
        for row in d['reserved_test']:
            p=actual(m.t.r.tensor(m.t.r.read(row['path']))[None])[0][0]
            m.t.r.c.rgb_image(p).save(dest/(row['id']+'_reserved_integer720.png'))
    write_json(dest/'review.json',results)
    print('MINIMUM_AREA_REVIEW',dest,results['validation']['mean'],flush=True)


if __name__=='__main__':main()
