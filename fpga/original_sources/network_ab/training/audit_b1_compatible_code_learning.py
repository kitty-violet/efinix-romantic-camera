"""Explain direct-code learning progress without inferring a capacity bound."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .runtime import write_json
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    torch.set_num_threads(2);root=Path(a.run);conf=json.loads((root/'config.json').read_text(encoding='utf-8'));args=conf['args']
    rows={r['id']:r for r in m.load_data(args['data'])['training']};xs=[]
    for s in conf['specs']:
        xx,yy=s['xy'];w,h=s['wh'];v=m.t.r.read(rows[s['row']['id']]['path'])[yy:yy+h,xx:xx+w]
        xs.append(torch.from_numpy(v.transpose(2,0,1).copy()).float()/128-1)
    x=torch.stack(xs);truth=torch.load(args['latent'],weights_only=True)['features'].float();cut=lambda z:z[...,10:-10,10:-10]
    results={}
    for label,path in [('initial',args['checkpoint']),('learned',str(root/'local_001200.pt'))]:
        _,net=load(path);net.__class__=ReadoutB1;nodes=net(x,all_nodes=True)
        loss=F.smooth_l1_loss(cut(nodes['res3_add']),cut(truth),beta=1.);loss.backward()
        results[label]=dict(loss=float(loss),features=[],layers=[])
        for c in range(32):
            z=cut(nodes['res3_add'])[:,c].detach();t=cut(truth)[:,c]
            results[label]['features'].append(dict(channel=c,mae=float((z-t).abs().mean()),
                at_limit=float(((z<=-128)|(z>=127)).float().mean()),std=float(z.std()),target_std=float(t.std()),mean=float(z.mean()),target_mean=float(t.mean())))
        for row in net.plan['nodes']:
            k=row['id'].replace('.','__')
            if k not in net.weight:continue
            w=net.weight[k];scale=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
            g=w.grad
            results[label]['layers'].append(dict(layer=k,gradient=0. if g is None else float(g.norm()),
                zero_gradient_rows=len(w) if g is None else int((g.flatten(1).abs().sum(1)==0).sum()),
                active_rows=len(w),limit_fraction=float((w.abs()>=127*scale).float().mean()),
                activation_std=float(nodes[row['id']].detach().std())))
        print(label,'loss',float(loss),'feature_mae',[round(z['mae'],2) for z in results[label]['features']],flush=True)
        print(label,'layers',results[label]['layers'],flush=True)
    write_json(root/'learning-path-audit.json',results)


if __name__=='__main__':main()
