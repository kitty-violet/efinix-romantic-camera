"""Measure whether the old flatness loss penalizes ideal hardware targets."""
import json
import numpy as np
from PIL import Image
from .train_luma_regions import ROOT,RUN,LEVELS
from .probe_class_rank import decode
from .runtime import write_json,file_hash


def labs(t):
    r,g,b=np.moveaxis(t.astype(np.float32),-1,0)
    cs=[.25*r+.5*g+.25*b,.5*r-.5*b,-.25*r+.5*g-.25*b]
    return np.stack([np.abs(c[...,None]-np.asarray(v)).argmin(-1) for c,v in zip(cs,LEVELS)],-1).astype(np.uint8)


def main():
    root=ROOT/'runs'/RUN;out=root/'region-loss-contract-audit';out.mkdir(exist_ok=False)
    data=json.loads((root/'native-mid-fp32/data.json').read_text(encoding='utf-8'));groups={}
    for r in data['pairs']:groups.setdefault(r['source'],[]).append(r)
    records=[]
    for source,rs in groups.items():
        r=rs[len(rs)//2];t=np.asarray(Image.open(r['current_target']).convert('RGB')).astype(np.float32);l=labs(t);q=decode(l).astype(np.float32);vals=[]
        for d in [1,3,7]:
            for axis in [0,1]:
                aa=[slice(None)]*3;bb=aa.copy();aa[axis]=slice(None,-d);bb[axis]=slice(d,None);aa=tuple(aa);bb=tuple(bb)
                td=np.max(np.abs(t[aa]-t[bb]),-1);mask=(td<4)[32:-32,32:-32]
                err=np.maximum(np.mean(np.abs(q[aa]-q[bb]),-1)-td-1,0)[32:-32,32:-32]
                same=np.all(l[aa]==l[bb],-1)[32:-32,32:-32]
                assert not np.any(err[same]>0)
                vals.append(dict(offset=d,axis=axis,ideal_old_flat_rgb=float(err[mask].mean()) if mask.any() else 0,
                    ideal_penalized_fraction=float((err[mask]>0).mean()) if mask.any() else 0,ideal_corrected_flat_rgb=0.0))
        records.append(dict(source=source,target=r['current_target'],target_sha256=file_hash(r['current_target']),tests=vals))
    mean=float(np.mean([r['ideal_old_flat_rgb'] for q in records for r in q['tests']]));fraction=float(np.mean([r['ideal_penalized_fraction'] for q in records for r in q['tests']]))
    write_json(out/'report.json',dict(status='IDEAL_TARGET_LOSS_CONTRADICTION_MEASURED',mean_old_flat_rgb=mean,mean_old_penalized_fraction=fraction,corrected_flat_rgb=0.,records=records,source_sha256=file_hash(__file__)))
    print('REGION_LOSS_CONTRACT',mean,fraction,flush=True)


if __name__=='__main__':main()
