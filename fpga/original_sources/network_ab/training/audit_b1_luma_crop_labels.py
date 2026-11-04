"""Check that one source pixel gets the same Y class in different crops.

The current label function averages antialiased target pixels inside each
region in the supplied tensor. Cropping may change that mean at a threshold.
Measure this using the actual PyTorch label function, not an assumed cause.
"""
import argparse,random
from pathlib import Path
import numpy as np,torch
from .train_b1_luma_chroma_interfaces import luminance_codes
from .qat_b1_deploy_preview import Round
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);rng=random.Random(2026100727);rnd=Round();data=m.load_data(a.data);records=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2]
        tensor=lambda path:torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
        target=tensor(row['target']);lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        full=luminance_codes(target,lab,rnd)//81;hard=luminance_codes(tensor(row['hard_region_target']),lab,rnd)//81
        trials=[]
        for _ in range(40):
            xx=rng.randrange(1280-320+1);yy=rng.randrange(720-320+1);sl=(slice(None),slice(yy,yy+320),slice(xx,xx+320))
            cropped=luminance_codes(target[:,:,yy:yy+320,xx:xx+320],lab[sl],rnd)//81
            # Only pixels that participate in the real cropped loss.
            d=(cropped-full[sl])[...,40:-40,40:-40];trials.append(dict(xy=[xx,yy],changed=float((d!=0).float().mean()),bright=float((d>0).float().mean()),max_step=int(d.abs().max())))
        record=dict(id=row['id'],full_soft_vs_hard_fraction=float((full!=hard).float().mean()),
            crop_class_change_mean=sum(r['changed'] for r in trials)/len(trials),crop_class_change_max=max(r['changed'] for r in trials),trials=trials)
        records.append(record);print('LUMA_CROP_LABELS',row['id'],record['crop_class_change_mean'],record['crop_class_change_max'],record['full_soft_vs_hard_fraction'],flush=True)
        write_json(a.out,dict(records=records,complete=False,source_sha256=file_hash(__file__)))
    write_json(a.out,dict(records=records,complete=True,train_only=True,source_sha256=file_hash(__file__),label_function_sha256=file_hash(Path(__file__).with_name('train_b1_luma_chroma_interfaces.py'))))


if __name__=='__main__':main()
