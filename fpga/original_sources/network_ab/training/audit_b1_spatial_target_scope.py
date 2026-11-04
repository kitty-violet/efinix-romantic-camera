"""Measure B1's exact local dependency and the offline region-mean scope.

This is a target-design diagnostic, not a natural-image impossibility proof.
The teacher's own receptive field is not assumed equal to B1's, and replacing
a target with local averages is NOT proposed or performed here.
"""
import argparse,json
from functools import lru_cache
from pathlib import Path
import numpy as np, torch
from .probe_b1_encoder_gauge import load
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def dependency(nodes):
    byid={n['id']:n for n in nodes}
    @lru_cache(None)
    def span(name,i):
        if name=='input':return i,i
        r=byid[name]
        if r['kind']=='nearest_x2':return span(r['inputs'][0],i//2)
        if r['kernel']:
            indices=range(i*r['stride']-r['kernel']//2,i*r['stride']+r['kernel']//2+1)
            pairs=[span(r['inputs'][0],j) for j in indices]
        else:pairs=[span(k,i) for k in r['inputs']]
        return min(v[0] for v in pairs),max(v[1] for v in pairs)
    return span


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    _,net=load(a.checkpoint);span=dependency(net.plan['nodes'])
    phases=[dict(phase=i,relative_span=[z-(320+i) for z in span('rgb_head',320+i)],
                 size=span('rgb_head',320+i)[1]-span('rgb_head',320+i)[0]+1) for i in range(4)]
    data=m.load_data(a.data);samples=[];numeric=[]
    for source in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==source];row=rows[len(rows)//2]
        parent=m.t.r.read(row['parent_target']).astype(float)
        labels=m.load_region(row['region']).astype(np.int64)
        n=int(labels.max())+1;count=np.bincount(labels.ravel(),minlength=n)
        sums=np.stack([np.bincount(labels.ravel(),weights=parent[...,c].ravel(),minlength=n) for c in range(3)],1)
        means=sums/count[:,None].clip(1)
        for yy in range(64,680,80):
            for xx in range(64,1240,80):
                ya,yb=span('rgb_head',yy);xa,xb=span('rgb_head',xx)
                lab=int(labels[yy,xx]);mask=labels[ya:yb+1,xa:xb+1]==lab
                local=parent[ya:yb+1,xa:xb+1][mask].mean(0)
                samples.append(dict(id=row['id'],xy=[xx,yy],region=lab,area=int(count[lab]),
                    mean_support_outside_B1_window_fraction=1-float(mask.sum()/count[lab]),
                    same_region_local_mean_vs_full_mean_mae=float(np.abs(local-means[lab]).mean())))
        # Confirm on actual integer arithmetic that remote pixels cannot
        # affect a centre2x2 whose complete dependency window is preserved.
        source_rgb=m.t.r.read(row['path'])[160:352,480:672]
        x=torch.from_numpy(source_rgb.transpose(2,0,1).copy())[None].float()/128-1
        y=x.clone();yy=xx=96
        ya=span('rgb_head',yy)[0];yb=span('rgb_head',yy+1)[1]
        xa=span('rgb_head',xx)[0];xb=span('rgb_head',xx+1)[1]
        gen=torch.Generator().manual_seed(2026100809)
        remote=torch.randint(-128,128,y.shape,generator=gen).float()/128
        remote[...,ya:yb+1,xa:xb+1]=y[...,ya:yb+1,xa:xb+1]
        p=net(x)[0];q=net(remote)[0]
        error=float((p[...,yy:yy+2,xx:xx+2]-q[...,yy:yy+2,xx:xx+2]).abs().max()*128)
        assert error==0
        numeric.append(dict(id=row['id'],centre_error_byte=error,preserved_box=[xa,ya,xb+1,yb+1]))
    outside=np.array([r['mean_support_outside_B1_window_fraction'] for r in samples])
    delta=np.array([r['same_region_local_mean_vs_full_mean_mae'] for r in samples])
    result=dict(phases_720=phases,numeric_locality_checks=numeric,samples=samples,
        summary=dict(points=len(samples),points_with_remote_region_support=float((outside>0).mean()),
            mean_remote_fraction=float(outside.mean()),mean_local_vs_global_colour_delta=float(delta.mean()),
            p95_local_vs_global_colour_delta=float(np.percentile(delta,95)),over8=float((delta>8).mean())),
        checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),source_sha256=file_hash(__file__),
        target_files_modified=False,scope='Dependency box upper bound includes all possible convolution paths. Region-mean scope is measured on the offline teacher RGB, not a proof of irreducible source-to-target error.')
    out=Path(a.out);out.mkdir(exist_ok=False);write_json(out/'review.json',result)
    print('SPATIAL_TARGET_SCOPE',phases,result['summary'],'integer_locality0',flush=True)


if __name__=='__main__':main()
