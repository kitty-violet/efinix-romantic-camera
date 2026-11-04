"""TRAIN-only test of joint patch-shape distribution gradients."""
import argparse,json,shutil,copy
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decision_slope_current import harden
from .b1_local_patch_shape_distribution import patch_shape_distribution,regression
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);regression();_,net=load(a.checkpoint);nets={'actual':net}
    for name,gain in (('half_slope',.5),('double_slope',2)):
        n=copy.deepcopy(net);harden(n,gain);nets[name]=n
    rows={r['id']:r for r in m.load_data(a.data)['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];records=[]
    for item in (mined[1],mined[5],mined[6]):
        row=rows[item['row']['id']];xx,yy=item['xy'];w,h=item['wh'];cut=lambda z:z[...,40:-40,40:-40]
        tensor=lambda k:torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        x=tensor('path');t=cut(tensor('target'))
        with torch.no_grad():
            imgs={k:cut(n(x)[0]) for k,n in nets.items()}
            imgs['blur5']=F.avg_pool2d(F.pad(imgs['actual'],(2,2,2,2),mode='replicate'),5,1);imgs['target']=t
        p=imgs['actual'].clone().requires_grad_();loss=patch_shape_distribution(p,t);grad=torch.autograd.grad(loss,p)[0]
        records.append(dict(id=row['id'],losses={k:float(patch_shape_distribution(v,t)) for k,v in imgs.items()},
            gradients_negative_prefers_change={k:float((grad*(v-p.detach())).sum()) for k,v in imgs.items()},gradient_norm=float(grad.norm())))
        print('PATCH_DIRECTION',records[-1],flush=True)
    out=Path(a.out);out.mkdir(exist_ok=False)
    for f in (Path(__file__),Path(__file__).with_name('b1_local_patch_shape_distribution.py')):shutil.copy2(f,out/f.name)
    write_json(out/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),scope='RGB gradients and finite proposals, fixed TRAIN crops, not complete parameter update.'))


if __name__=='__main__':main()
