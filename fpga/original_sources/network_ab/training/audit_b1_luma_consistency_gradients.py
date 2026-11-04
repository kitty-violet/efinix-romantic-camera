"""Measure actual TRAIN flow/phase gradients versus14Y+chroma supervision.

Read fixed centering means from the real run's optimizer, not validation. This
reports gradient interactions, not a proof that fixed-weight videos flicker.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_luma_chroma_interfaces import luminance_codes,chroma_objective
from .train_b1_direct_luma_interfaces import luma_decision_loss
from .train_b1_ordered_distance_rims import distance_objective
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .train_b1_real_region_texture import align_pair
from .train_b1_style_recovery import read_flow,warp
from .runtime import write_json,file_hash
from .audit_b1_rim_tolerance import rim_terms
from .b1_active_luma_interface import active_interface_loss
from . import train_b1_minimum_area as m


def vector(loss,net,means,keys):
    pp=[v for k in keys for v in (net.weight[k],net.bias[k])]
    gradients=torch.autograd.grad(loss,pp,retain_graph=True,allow_unused=True);result=[]
    for i,k in enumerate(keys):
        w,b=pp[2*i:2*i+2];gw,gb=gradients[2*i:2*i+2]
        gw=torch.zeros_like(w) if gw is None else gw;gb=torch.zeros_like(b) if gb is None else gb
        mu=means[k].to(w);mu=mu[:,None,None,None] if w.shape[1]==1 else mu[None,:,None,None]
        result.extend(((gw-gb[:,None,None,None]*mu).flatten(),gb.flatten()))
    return torch.cat(result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=OrdinalBandB1
    keys=[k for k in net.weight if not k.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) and k not in ('down1__dw','down2__dw')]
    for k in net.weight:net.weight[k].requires_grad_(k in keys);net.bias[k].requires_grad_(k in keys)
    means={g['node_key']:g['input_mean'].cpu() for g in state['optimizer']['param_groups'] if 'node_key' in g}
    data=m.load_data(a.data);by_path={r['path']:r for r in data['training']};pairs=[]
    for source in sorted({p['source'] for p in data['pairs']}):
        rows=[p for p in data['pairs'] if p['source']==source];pairs.append(rows[len(rows)//2])
    records=[];cut=lambda q:q[...,40:-40,40:-40]
    for pair in pairs:
        xx,yy=480,200;h=w=320;sl=np.s_[yy:yy+h,xx:xx+w]
        xs=[];ts=[];labs=[]
        for side in ('current','next'):
            row=by_path[pair[side]]
            xs.append(torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
            ts.append(torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy()).float()/128-1)
            labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        x=torch.stack(xs);t=torch.stack(ts);labels=torch.stack(labs)
        ff,vv=read_flow(pair['flow']);ff=ff[sl];vv=vv[sl].copy();gy,gx=np.mgrid[:h,:w]
        vv*=((gx+ff[:,:,0]>=40)&(gx+ff[:,:,0]<w-40)&(gy+ff[:,:,1]>=40)&(gy+ff[:,:,1]<h-40))
        flow=torch.from_numpy(ff.copy()).permute(2,0,1)[None];valid=torch.from_numpy(vv)[None,None]
        code=luminance_codes(t,labels,net.round);t,labels,_,fraction=align_pair(x,t,labels,code,flow,valid)
        code=luminance_codes(t,labels,net.round);truth=joint_interface_fields(code)[:,:14]
        nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128;field=nodes['up1_low_dw'][:,:14,...,10:-10,10:-10]
        full,band=distance_objective(field,truth[...,10:-10,10:-10])
        with torch.no_grad():want=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,18)),'up1_low_dw')['up1_high_dw'][:,:14]
        kl,near=luma_decision_loss(nodes['up1_high_dw'][:,:14,...,20:-20,20:-20],want[...,20:-20,20:-20])
        rank=(F.relu(torch.diff(field,dim=1)).amax(1)/127).square().mean()
        colour,_=chroma_objective(cut(p),cut(t),cut(labels));primary=4*full+2*band+16*kl+8*near+10*rank+colour
        shifted=net(torch.roll(x[:1],(1,1),(-2,-1)))[0]
        phase=1.5*(cut(p[:1])-cut(torch.roll(shifted,(-1,-1),(-2,-1)))).abs().mean()
        confidence=valid*torch.exp(-((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)/.08)**2)
        temporal=2*(cut((p[1:]-warp(p[:1],flow)).abs())*cut(confidence)).sum()/(3*cut(confidence).sum()+1e-8)
        rim=4*rim_terms(cut(p),cut(t),cut(labels),2)[0]
        shared,active_fraction=active_interface_loss(field,truth[...,10:-10,10:-10]);shared=4*shared
        gv={name:vector(loss,net,means,keys) for name,loss in (('primary',primary),('phase',phase),('temporal',temporal),('tight_rim',rim),('active_interface',shared))}
        rec=dict(source=Path(pair['source']).name,xy=[xx,yy],aligned_target_fraction=float(fraction),active_interface_fraction=float(active_fraction),
            losses={name:float(loss.detach()) for name,loss in (('primary',primary),('phase',phase),('temporal',temporal),('tight_rim',rim),('active_interface',shared))},
            centred_gradient_norm={name:float(g.norm()) for name,g in gv.items()},
            cosine_to_primary={name:float(F.cosine_similarity(gv['primary'],gv[name],dim=0)) for name in ('phase','temporal','tight_rim','active_interface')})
        records.append(rec);write_json(a.out,dict(records=records,training_sources_only=True,weight_dynamics_not_video_metric=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__)))
        print('LUMA_CONSISTENCY_AUDIT',rec,flush=True)
    print('LUMA_CONSISTENCY_AUDIT_COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
