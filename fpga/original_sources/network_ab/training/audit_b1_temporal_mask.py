"""CPU audit: target disagreement must not hide stable-source flicker."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def main():
    cv2.setNumThreads(2)
    root=Path(__file__).resolve().parents[2]/'model-training-workspace/runs/b1-minimum-area-20261006'
    data=json.loads((root/'data.json').read_text(encoding='utf-8'));records=[]
    yy,xx=np.mgrid[:720,:1280].astype(np.float32)
    for pair in data['pairs']:
        with np.load(pair['flow']) as z:
            f=cv2.resize(z['backward'].astype(np.float32),(1280,720))*2
            valid=cv2.resize(z['mask'].astype(np.float32),(1280,720),interpolation=cv2.INTER_NEAREST)
        differences={}
        for name,ka,kb in (('source','current','next'),('target','current_target','next_target')):
            a=np.asarray(Image.open(pair[ka]).convert('RGB'),dtype=np.float32)
            b=np.asarray(Image.open(pair[kb]).convert('RGB'),dtype=np.float32)
            wa=cv2.remap(a,xx+f[...,0],yy+f[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            differences[name]=abs(b-wa).mean(2)
        reliable=(valid>0)&(differences['source']<8)
        old=np.exp(-(differences['target']/10.24)**2)
        new=np.exp(-(differences['source']/10.24)**2)
        records.append(dict(source=pair['source'],frame=pair['current_index'],
            reliable_pixels=int(reliable.sum()),old_mean_weight_on_source_stable=float(old[reliable].mean()),
            source_mean_weight_on_source_stable=float(new[reliable].mean()),
            source_stable_pixels_target_weight_below_point1=float((old[reliable]<.1).mean())))
    result=dict(records=records,meaning='Original temporal weighting excludes some stable-source pixels when independently generated target regions disagree',
                training_only=True,visual_acceptance=False)
    (root/'temporal-mask-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TEMPORAL_MASK_AUDIT', {k:float(np.mean([r[k] for r in records])) for k in records[0] if k not in ('source','frame','reliable_pixels')},flush=True)


if __name__=='__main__':main()
