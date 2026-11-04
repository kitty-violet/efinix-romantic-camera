"""Quantify first-step optimization blocked by fixed coefficient ranges.

Counts of saturated weights alone are not a proof of a bottleneck. This
diagnostic reports the removed tentative update in the optimizer's centred
coordinates. It is a local gradient estimate, not an actual quality gain.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .whitened_b1_optimizer import WhitenedB1Adam,patch_statistics,transform
from .train_b1_quantum_regions import RowScaledAdam
from .b1_geometry_first_objective import geometry_first
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);_,net=load(a.checkpoint);net.__class__=ReadoutB1
    rows={r['id']:r for r in m.load_data(a.data)['training']};items=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];xs=[];ts=[];ls=[]
    for s in (items[1],items[5],items[6]):
        row=rows[s['row']['id']];xx,yy=s['xy'];w,h=s['wh']
        def read(k):return torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float()/128-1
        xs.append(read('path'));ts.append(read('target'));ls.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)))
    x,t,lab=torch.stack(xs),torch.stack(ts),torch.stack(ls);stats=patch_statistics(net,x,4096);groups=[]
    for k,w in net.weight.items():
        scale=(getattr(net,k+'_m')/131072).float().flatten()
        groups.extend([dict(params=[w],lr=.0002,row_gain=(scale*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.008)])
    opt=WhitenedB1Adam(groups,net,stats,True);p=net(x)[0];cut=lambda z:z[...,40:-40,40:-40]
    loss,_=geometry_first(cut(p),cut(t),cut(lab));loss.backward()
    centred={k:(w.grad.flatten(1)-net.bias[k].grad[:,None]*stats[k]['mean']).clone() for k,w in net.weight.items()}
    opt.prepare_gradients();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
    before={k:w.detach().clone() for k,w in net.weight.items()}
    RowScaledAdam.step(opt);records=[]
    with torch.no_grad():
        for group in opt.param_groups:
            if 'node_key' not in group:continue
            k=group['node_key'];w=net.weight[k];old=before[k]
            delta=transform((w-old).flatten(1),group['input_whitener']).reshape_as(w)
            scale=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
            proposed=old+delta;clipped=proposed.maximum(-127*scale).minimum(127*scale);removed=proposed-clipped
            g=centred[k].reshape_as(w);blocked_gain=float(-(g*removed).sum());proposed_gain=float(-(g*delta).sum())
            record=dict(layer=k,proposed_linear_gain=proposed_gain,blocked_linear_gain=blocked_gain,
                fraction_clipped=float((removed!=0).float().mean()),max_removed_in_codes=float((removed/scale).abs().max()),
                max_effective_coefficient=float((127*scale).max()))
            if k in ('down1__dw','down2__dw'):
                record['centre_weight']=old[:,0,1,1].tolist();record['allowed_magnitude']=(127*scale.flatten()).tolist()
            records.append(record)
    positive=sum(max(r['proposed_linear_gain'],0) for r in records);blocked=sum(max(r['blocked_linear_gain'],0) for r in records)
    out=Path(a.out);out.mkdir(exist_ok=False);result=dict(records=records,positive_proposed_gain=positive,positive_blocked_gain=blocked,
        ratio=blocked/max(positive,1e-12),checkpoint_sha256=file_hash(a.checkpoint),scope='Fresh optimizer first step, local proxy-gradient estimate, not achieved quality.')
    write_json(out/'results.json',result);print('PROJECTED_UPDATE_AUDIT',result,flush=True)


if __name__=='__main__':main()
