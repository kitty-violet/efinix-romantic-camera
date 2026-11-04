"""Measure missing-core region coverage in fixed TRAIN crops and seen validation."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from .b1_all_region_means import core_mask,regression
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def coverage(labels):
    flat=labels.flatten();_,idx=torch.unique(flat,return_inverse=True)
    area=torch.bincount(idx);core=core_mask(labels).flatten()
    inside=torch.bincount(idx[core],minlength=len(area));empty=inside==0
    return dict(regions=len(area),regions_without_core=int(empty.sum()),
        pixels_in_regions_without_core=int(area[empty].sum()),pixels=flat.numel(),
        fraction_region_mean_weight_missing=float(area[empty].sqrt().sum()/area.sqrt().sum()),
        fraction_boundary_pixels=float((~core).float().mean()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);checks=regression();data=m.load_data(a.data)
    rows={r['id']:r for r in data['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];results=[]
    for item in (mined[1],mined[5],mined[6]):
        row=rows[item['row']['id']];x,y=item['xy'];w,h=item['wh']
        labels=m.load_region(row['region'])[y+40:y+h-40,x+40:x+w-40]
        results.append(dict(id=row['id'],split='fixed_training_crop_inner240',**coverage(torch.from_numpy(labels.astype(np.int64))[None])))
    for row in data['validation']:
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        results.append(dict(id=row['id'],split='seen_validation_full720',**coverage(labels)))
    out=Path(a.out);out.mkdir(exist_ok=False)
    for source in (Path(__file__),Path(__file__).with_name('b1_all_region_means.py')):shutil.copy2(source,out/source.name)
    write_json(out/'results.json',dict(records=results,regression=checks,data_sha256=file_hash(a.data),
        scope='Missing region-mean supervision only; pixel and complete-profile losses already cover these pixels.'))
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
