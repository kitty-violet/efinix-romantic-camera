"""Joint missing-edge/spurious-edge and target-interior diagnostics.

These are supporting measurements at native720, never style acceptance. All
model outputs come from reloaded coefficient bytes, including residual fields.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .review_b1_minimum_area import region_diagnostics
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def edges(rgb,threshold=10):
    x=rgb.astype(np.float32);g=np.zeros(x.shape[:2],np.float32)
    g[:,1:]=np.maximum(g[:,1:],np.abs(x[:,1:]-x[:,:-1]).max(2))
    g[1:]=np.maximum(g[1:],np.abs(x[1:]-x[:-1]).max(2))
    return g>threshold


def compare(rgb,target):
    p,t=edges(rgb),edges(target)
    dt=cv2.distanceTransform((~t).astype(np.uint8),cv2.DIST_L2,5)
    dp=cv2.distanceTransform((~p).astype(np.uint8),cv2.DIST_L2,5)
    result=dict(predicted_edge_pixels=int(p.sum()),target_edge_pixels=int(t.sum()))
    for radius in (2.,4.):
        precision=float((dt[p]<=radius).mean()) if p.any() else 0.
        recall=float((dp[t]<=radius).mean()) if t.any() and p.any() else 0.
        result[str(radius)]=dict(precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-9))
    return result


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100691);cv2.setNumThreads(2);d=m.load_data();results=[]
    test=np.zeros((128,128,3),np.uint8);test[:,64:]=200
    checks={name:compare(v,test) for name,v in [('exact',test),('flat',test*0),('noise',np.random.default_rng(1).integers(0,255,test.shape,dtype=np.uint8))]}
    assert checks['exact']['2.0']['f1']==1 and checks['flat']['2.0']['f1']==0
    assert checks['noise']['2.0']['precision']<.1
    for path in a.checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
        if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()));records=[]
        for row in d['validation']:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];rgb=(actual(x*128)[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
            target=m.t.r.read(row['target']);labels=m.load_region(row['region'])
            records.append(dict(id=row['id'],edges=compare(rgb,target),interior=region_diagnostics(rgb,target,labels)))
        results.append(dict(checkpoint=path,sha256=file_hash(path),records=records))
        print('REGION_SHAPE_AUDIT',Path(path).parent.name,
              'edge_f1',round(np.mean([r['edges']['2.0']['f1'] for r in records]),4),
              'interior_noise',round(np.mean([r['interior']['interior_transition_over6_fraction'] for r in records]),4),flush=True)
        del net,actual;torch.cuda.empty_cache()
    write_json(a.output,dict(results=results,synthetic_rejection_checks=checks,visual_acceptance=False,postfilter=False))


if __name__=='__main__':main()
