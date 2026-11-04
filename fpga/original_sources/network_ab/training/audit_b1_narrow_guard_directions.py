"""TRAIN image-gradient checks before adding the RGB-envelope guard.

These gradients are not optimizer parameter steps or proof of video quality.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_narrow_recall_native import objective
from .b1_local_rgb_envelope import local_envelope,core_error_variation
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    byid={r['id']:r for r in m.load_data(a.data)['training']};specs=json.loads(Path(a.mining).read_text('utf-8'))['records']
    _,net=load(a.checkpoint);net.requires_grad_(False);records=[]
    for k in (1,5,6):
        s=specs[k];row=byid[s['row']['id']];xx,yy=s['xy'];w,h=s['wh']
        read=lambda key:torch.from_numpy(m.t.r.read(row[key])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        with torch.no_grad():p=net(read('path'))[0][...,40:-40,40:-40].clone()
        p.requires_grad_();t=read('target')[...,40:-40,40:-40]
        lab=torch.from_numpy(m.load_region(row['region'])[yy+40:yy+h-40,xx+40:xx+w-40].astype(np.int64))[None]
        loss,_=objective(p,t,lab);guard,_=local_envelope(p,t,lab);core=core_error_variation(p,t,lab)
        grads=[torch.autograd.grad(v,p,retain_graph=True)[0] for v in (loss,8*guard,8*core)]
        total=sum(grads);blur=F.avg_pool2d(F.pad(p.detach(),(2,2,2,2),mode='replicate'),5,1)
        rec=dict(id=row['id'],values=[float(v.detach()) for v in (loss,8*guard,8*core)],
            image_gradient_norm=[float(g.norm()) for g in grads],
            main_guard_cosine=float((grads[0]*grads[1]).sum()/(grads[0].norm()*grads[1].norm()).clamp_min(1e-20)),
            target_direction=float((total*(t-p.detach())).sum()),
            blur_direction=float((total*(blur-p.detach())).sum()))
        assert np.isfinite(rec['values']).all() and np.isfinite(rec['image_gradient_norm']).all()
        records.append(rec)
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),image_gradients_only=True,visual_acceptance=False))
    print('NARROW_GUARD_TRAIN_DIRECTIONS',records,flush=True)


if __name__=='__main__':main()
