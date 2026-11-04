"""Check whether clipped colour-lattice inversion creates large latent jumps."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .probe_b1_encoder_gauge import load
from .train_b1_ternary_regions import region_codes
from .audit_b1_target_chain import region_means,adjacency
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);setup=state['binary_coverage720']
    coordinates=np.array([(y,o,g) for y in range(15) for o in range(9) for g in range(9)],np.float32)
    unclip=coordinates@np.asarray(setup['matrix'],np.float32).T+np.asarray(setup['bias'],np.float32)+128
    palette=unclip.clip(0,255);pk=tuple(map(tuple,palette.tolist()))
    rgb_distance=np.linalg.norm(palette[:,None]-palette[None],axis=-1)
    latent_distance=np.abs(coordinates[:,None]-coordinates[None]).sum(-1)
    mask=np.triu((rgb_distance<=6)&(latent_distance>=4),1)
    records=[];data=m.load_data(a.data)
    chosen=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];chosen.append(rows[len(rows)//2])
    for row in chosen+data['validation']:
        labels=m.load_region(row['region']).astype(np.int64);rgb=m.t.r.read(row['target']);means,count=region_means(rgb,labels)
        codes=region_codes(row['target'],row['region'],pk);by_region=np.zeros(len(count),np.int64);by_region[labels.ravel()]=codes.ravel()
        aa,bb,length=adjacency(labels);ca,cb=by_region[aa],by_region[bb]
        near=np.linalg.norm(palette[ca]-palette[cb],axis=1)<=6
        jump=np.abs(coordinates[ca]-coordinates[cb]).sum(1)>=4
        records.append(dict(id=row['id'],boundary_length=int(length.sum()),near_rgb_large_latent_fraction=float(length[near&jump].sum()/max(1,length.sum())),
            clipped_code_area_fraction=float((np.any(unclip[by_region]!=palette[by_region],axis=1)*count).sum()/count.sum()),
            colour_fit_mae=float((np.abs(means-palette[by_region]).mean(1)*count).sum()/count.sum())))
    result=dict(unique_colours=len(np.unique(palette,axis=0)),palette_entries=len(palette),clipped_entries=int(np.any(unclip!=palette,axis=1).sum()),
        near_colour_large_code_pairs=int(mask.sum()),records=records,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),
        palette_or_training_changed=False,visual_acceptance=False)
    write_json(Path(a.out),result);print('PALETTE_INVERSE_AUDIT',result,flush=True)


if __name__=='__main__':main()
