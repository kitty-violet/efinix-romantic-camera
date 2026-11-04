"""Fixed TRAIN output-space direction checks for relative RGB profiles."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .b1_geometry_first_objective import geometry_first
from .b1_relative_rgb_profiles import relative_profiles,regression
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','data','mining','out'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();torch.set_num_threads(2);regression();_,net=load(a.checkpoint)
    rows={r['id']:r for r in m.load_data(a.data)['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];records=[]
    for spec in (mined[1],mined[5],mined[6]):
        row=rows[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        def read(k):return torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        x,t=read('path'),read('target');lab=torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64))[None]
        with torch.no_grad():p=net(x)[0][...,40:-40,40:-40]
        t=t[...,40:-40,40:-40];lab=lab[...,40:-40,40:-40]
        blur=F.avg_pool2d(F.pad(p,(2,2,2,2),mode='replicate'),5,1)
        records_here={}
        for name,fn in [('base',lambda z:geometry_first(z,t,lab)[0]),('relative',lambda z:relative_profiles(z,t,lab)[0]),
                        ('combined',lambda z:geometry_first(z,t,lab)[0]+relative_profiles(z,t,lab)[0])]:
            pred=p.detach().requires_grad_();loss=fn(pred);grad=torch.autograd.grad(loss,pred)[0]
            with torch.no_grad():
                records_here[name]=dict(loss=float(loss.detach()),blur5_loss=float(fn(blur)),target_loss=float(fn(t)),
                    gradient_norm=float(grad.norm()),towards_target=float((grad*(t-p)).sum()),towards_blur=float((grad*(blur-p)).sum()))
        records.append(dict(id=row['id'],values=records_here));print('RELATIVE_DIRECTION',row['id'],records_here,flush=True)
    out=Path(a.out);out.mkdir(exist_ok=False)
    write_json(out/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),scope='Output-space directions only, not a guarantee on parameter updates or unseen scenes.'))


if __name__=='__main__':main()
