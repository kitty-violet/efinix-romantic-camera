"""TRAIN-only gradient/forward audit of final-decision supervision."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_ordered_distance_geometry import distance_fields
from .probe_b1_decoder_coverage import fragment
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_render_boundary import decision_objective
from .b1_render_boundary_objective import rendering_decision_loss,smooth_rank_loss,regression
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_ternary_regions import region_codes
from .fit_b1_distance_geometry import optimizer
from .centered_b1_optimizer import calibration_means
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);regression();state,net=load(a.checkpoint);net.__class__=OrdinalBandB1;data=m.load_data(a.data)
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    rows=[]
    for video in sorted({r['video'] for r in data['training']}):
        rr=[r for r in data['training'] if r['video']==video];rows.append(rr[len(rr)//2])
    records=[]
    for row in rows:
        sl=np.s_[200:520,480:800]
        x=torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None];truth=distance_fields(code,8)
        opt=optimizer(net,centred=True,means=calibration_means(net,x));params=[p for p in net.parameters() if p.requires_grad]
        nodes=net(x,all_nodes=True);full,band=distance_objective(nodes['up1_low_dw'][...,10:-10,10:-10],truth[...,10:-10,10:-10])
        old=16*full+8*band;new,v=decision_objective(net,nodes,truth,False);new=2*new
        gradients={};rank=10*rank_loss(nodes['up1_low_dw'][...,10:-10,10:-10]);soft_rank=10*smooth_rank_loss(nodes['up1_low_dw'][...,10:-10,10:-10])
        for name,loss in (('distance',old),('decisions',new),('rank',rank),('smooth_rank',soft_rank)):
            opt.zero_grad();loss.backward(retain_graph=True);opt.prepare_gradients()
            gradients[name]=torch.cat([p.grad.flatten() for p in params]);opt.prepared=False
        p=nodes['up1_high_dw'][:,:15]
        with torch.no_grad():
            wanted=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['up1_high_dw'][:,:15]
            pp=p[...,20:-20,20:-20];tt=wanted[...,20:-20,20:-20]
            opposite=(pp-63.5)*(tt-63.5)<0
            partial=(tt-63.5).abs()<24
            exact=sum(rendering_decision_loss(wanted,wanted));assert exact==0
        g,h=gradients['distance'],gradients['decisions'];record=dict(id=row['id'],old_loss=float(old.detach()),new_loss=float(new.detach()),
            rank_loss=float(rank.detach()),smooth_rank_loss=float(soft_rank.detach()),
            rank_centered_gradient_norm=float(gradients['rank'].norm()),smooth_rank_centered_gradient_norm=float(gradients['smooth_rank'].norm()),
            rank_vs_distance_cosine=float(F.cosine_similarity(gradients['rank'],g,dim=0)),
            **{k:float(z.detach()) for k,z in v.items()},distance_centered_gradient_norm=float(g.norm()),
            decision_centered_gradient_norm=float(h.norm()),gradient_cosine=float(F.cosine_similarity(g,h,dim=0)),
            target_decision_band_fraction=float(partial.float().mean()),wrong_side_fraction=float(opposite.float().mean()),
            wrong_side_at_clip_fraction=float((opposite&((pp==0)|(pp==127))).float().mean()),exact_target_decision_loss=float(exact))
        records.append(record);print('RENDER_DECISION_GRADIENT',record,flush=True)
    write_json(a.out,dict(records=records,training_only=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))


if __name__=='__main__':main()
