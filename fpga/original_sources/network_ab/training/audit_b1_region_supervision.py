"""Measure supervision coverage and effective-vs-quantized field mismatch."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_learned_colour import ColourB1
from .build_b1_ordered_palette import GROUPS,offsets
from .runtime import write_json,file_hash


def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    root=Path(__file__).resolve().parents[2]/'model-training-workspace/runs'
    out=root/'b1-minimum-area-20261006';data=json.loads((out/'data.json').read_text(encoding='utf-8'))
    records=[]
    for row in data['training']:
        with np.load(row['region']) as z:lab=z['labels'][::4,::4].astype(np.float32)
        hi=cv2.dilate(lab,np.ones((3,3),np.uint8));lo=cv2.erode(lab,np.ones((3,3),np.uint8))
        valid=hi==lo;ids=np.unique(lab);alive=np.unique(lab[valid])
        count=np.bincount(lab.astype(np.int64).ravel());covered=np.zeros(len(count),bool);covered[alive.astype(int)]=True
        small=(count*16<1024)&(count>0)
        records.append(dict(id=row['id'],valid_pixel_fraction=float(valid.mean()),region_count=len(ids),
            regions_without_interior=int(len(ids)-len(alive)),
            unpenalized_region_pixel_fraction=float(count[~covered].sum()/count.sum()),
            small_region_pixel_fraction=float(count[small].sum()/count.sum()),
            small_region_valid_fraction=float(valid[small[lab.astype(int)]].mean()) if small.any() else None))
    state_path=out/'flat-colour-fields-qat/candidate_003000.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    net=ColourB1(revive=False,source=root/'b1-identity-reconstruction-20261005/initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    clipping=[]
    for key,w in net.weight.items():
        scale=(getattr(net,key+'_m')/131072).float().flatten()
        clipping.append(dict(layer=key,outside_weight_code_range=float((w.abs()>127*scale[:,None,None,None]).float().mean()),
            max_code=float((w.abs()/scale[:,None,None,None]).max())))
    fields=[]
    with torch.no_grad():
        for row in data['training'][::30]:
            rgb=cv2.cvtColor(cv2.imread(row['path']),cv2.COLOR_BGR2RGB)[200:456,400:656]
            x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/128-1
            feature=net(x,all_nodes=True)['up1_low_dw'];key='up1_low_pw';w,b=net.weight[key],net.bias[key]
            scale=(getattr(net,key+'_m')/131072).float().flatten()
            qw=net.round(w/scale[:,None,None,None]).clamp(-127,127)*scale[:,None,None,None]
            qb=net.round(b/scale)*scale;off=offsets('cpu')
            raw=F.conv2d(feature,w,b);actual=F.conv2d(feature,qw,qb)
            fields.append(dict(id=row['id'],coordinate_mae=float((raw[:,:15]-actual[:,:15]).abs().mean()),
                               coordinate_max=float((raw[:,:15]-actual[:,:15]).abs().max())))
    result=dict(data_sha256=file_hash(out/'data.json'),checkpoint=str(state_path),coverage=records,
        coverage_mean={k:float(np.mean([r[k] for r in records])) for k in ('valid_pixel_fraction','unpenalized_region_pixel_fraction','small_region_valid_fraction')},
        weights=clipping,field_difference=fields,scope='Training supervision diagnosis, not visual acceptance')
    write_json(out/'region-supervision-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','weights')},ensure_ascii=False),flush=True)
    print('CLIPPED_ROWS',json.dumps([r for r in clipping if r['outside_weight_code_range']>0]),flush=True)


if __name__=='__main__':main()
