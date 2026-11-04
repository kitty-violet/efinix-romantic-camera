"""Audit final720 RGB regions from reloaded coefficients; never alter output.

Exact palette membership is separate from nearest-palette area diagnostics.
The latter is an analysis convention, not a filter applied to the video.
Also check spatial permutations: flat-pattern invariance alone is insufficient
when rounded partial counts are filtered before summation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components,boundary_mask
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .probe_b1_decoder_coverage import fragment
from .build_b1_ordered_palette import GROUPS
from .runtime import write_json,file_hash


def nearest_codes(rgb,palette):
    unique,inverse=np.unique(rgb.reshape(-1,3),axis=0,return_inverse=True)
    distance=((unique[:,None].astype(np.float32)-palette[None])**2).sum(2)
    best=distance.argmin(1)
    return best[inverse].reshape(rgb.shape[:2]),dict(unique_rgb=len(unique),
        off_palette_fraction=float((distance.min(1)[inverse]>0).mean()),
        palette_distance_mean=float(np.sqrt(distance.min(1)[inverse]).mean()))


@torch.inference_mode()
def spatial_permutation_audit(net):
    """Same-colour neighbouring patterns must not fabricate a colour seam."""
    records=[]
    for group,(start,end) in enumerate(GROUPS):
        p=((torch.arange(2**(end-start))[:,None]>>torch.arange(end-start)[None])&1).float()
        left,right=torch.where(p.sum(1)[:,None]==p.sum(1)[None])
        baseline=torch.zeros(len(left),16,4,4)
        baseline[:,:3]=127;baseline[:,7:9]=127;baseline[:,11:13]=127
        baseline[:,start:end]=p[left,:,None,None]*127
        ref=fragment(net,baseline,'up2_low_dw')['rgb_head']
        for geometry in ('vertical','diagonal'):
            z=baseline.clone()
            mask=torch.zeros(4,4,dtype=torch.bool)
            if geometry=='vertical':mask[:,2:]=True
            else:mask=torch.arange(4)[:,None]+torch.arange(4)[None]>=3
            z[:,start:end]=torch.where(mask[None,None],p[right,:,None,None]*127,z[:,start:end])
            actual=fragment(net,z,'up2_low_dw')['rgb_head'];delta=(actual-ref).abs()
            records.append(dict(group=group,geometry=geometry,patterns=len(left),
                max_rgb_error=float(delta.max()),changed_pixel_fraction=float((delta.amax(1)>0).float().mean()),
                affected_patterns=int((delta.flatten(1).amax(1)>0).sum())))
    return records


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--spatial-only',action='store_true');ap.add_argument('--train-fit',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    spatial=spatial_permutation_audit(net);print('SPATIAL_SAME_COLOUR',spatial,flush=True)
    blob=net.export();records=[]
    if not a.spatial_only:
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu')
        palette=(np.array(state['palette'],np.float32)*128+128).round().clip(0,255)
        pk=palette_key(torch.tensor(state['palette']))
        data=m.load_data();rows=data['validation'];overrides={};board=None
        if a.train_fit:
            assert state.get('seven_scene_fit_diagnostic') and len(state['fit_ids'])==7
            by_id={r['id']:r for r in data['training']};rows=[by_id[k] for k in state['fit_ids']]
            manifest=state['args']['class_manifest'];assert file_hash(manifest)==state['temporal_class_manifest_sha256']
            overrides=json.loads(Path(manifest).read_text(encoding='utf-8'))['overrides']
            folder=Path(a.out).with_suffix('');folder.mkdir(exist_ok=False)
            board=Image.new('RGB',(1440,304*len(rows)),'#17202a');draw=ImageDraw.Draw(board)
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
        for index,row in enumerate(rows):
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);output=np.array(m.t.r.c.rgb_image(nodes['rgb_head'][0]/128))
            codes,stats=nearest_codes(output,palette);regions=components(codes);area=np.bincount(regions.ravel())
            truth=region_class_map(row['target'],row['region'],pk)
            if row['id'] in overrides:
                entry=overrides[row['id']];assert file_hash(entry['path'])==entry['sha256']
                with np.load(entry['path']) as bundle:truth=bundle['classes'].copy()
            edge=boundary_mask(truth);interior=cv2.dilate(edge,np.ones((7,7),np.uint8))==0
            stats.update(id=row['id'],regions=len(area),minimum_area=int(area.min()),
                area_below256_pixel_fraction=float(area[area<256].sum()/area.sum()),
                boundary_density=float(boundary_mask(codes).mean()),
                interior_wrong_colour_fraction=float((palette[codes][interior]!=palette[truth][interior]).any(1).mean()),
                palette_colour_mae=float(np.abs(output.astype(float)-palette[truth]).mean()))
            records.append(stats);print('NATIVE_COLOUR_AUDIT',stats,flush=True)
            if a.train_fit:
                Image.fromarray(output).save(folder/(row['id']+'_actual720.png'))
                target=palette[truth].astype(np.uint8)
                for col,(pic,title) in enumerate(((rgb,'TRAIN原片'),(target,'本轮量化训练目标'),(output,'实际整数B1·训练拟合'))):
                    board.paste(Image.fromarray(pic).resize((480,270)),(col*480,index*304+34))
                    draw.text((col*480+4,index*304+4),row['id']+' | '+title,font=font,fill='white')
        if board:board.save(folder/'comparison.png')
    write_json(a.out,dict(checkpoint=a.checkpoint,checkpoint_sha256=file_hash(a.checkpoint),
        coefficient_sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),
        spatial_permutation_audit=spatial,records=records,
        resolution=[1280,720],actual_reloaded_coefficients=True,
        selection='seven TRAIN fitting scenes' if a.train_fit else 'six seen validation scenes',
        nearest_palette_for_area_analysis_only=True,postfilter_applied=False,visual_acceptance=False))


if __name__=='__main__':main()
