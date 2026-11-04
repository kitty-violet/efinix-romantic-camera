"""Trace actual current losses to the readout, using fixed TRAIN inverse labels."""
import argparse,json,types
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_relaxed_count_interfaces import RelaxedCountInterfaceTraining
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--targets',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,net=load(a.checkpoint);net.__class__=ReadoutB1
    data=m.load_data(a.data);rows={r['id']:r for r in data['training']};ref=torch.load(a.targets,map_location='cpu',weights_only=True);xs=[];ts=[];ls=[];cs=[]
    for s in ref['specs']:
        row=rows[s['row']['id']];xx,yy=s['xy'];w,h=s['wh'];sl=np.s_[yy:yy+h,xx:xx+w]
        def im(key):return torch.from_numpy(m.t.r.read(row[key])[sl].transpose(2,0,1).copy()).float()/128-1
        xs.append(im('path'));ts.append(im('target'));ls.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        with np.load(row['native_luma_codes']) as z:cs.append(torch.from_numpy(z['codes'][sl].astype(np.int64)))
    x,t,labels,code=map(torch.stack,(xs,ts,ls,cs));nodes=net(x,all_nodes=True);obj=object.__new__(RelaxedCountInterfaceTraining)
    obj.net=net;obj.nodes=nodes;obj.labels=labels;obj.class_labels=code;obj.args=types.SimpleNamespace(classification=2.);obj.motion_fraction=x.new_zeros(());obj.distance_cache=lambda c,g:ref['codes']
    cut=lambda z:z[...,40:-40,40:-40];loss,v=obj.additional_loss(cut(nodes['rgb_head']/128),cut(t),False)
    groups=dict(code=.5*v['ordered_distance_full']+.25*v['ordered_distance_band'],
        count=8*v['joint_luma_count_full']+4*v['joint_luma_count_boundary'],
        rendered_pixel=4*v['rendered_luma_pixel'],rendered_profile=4*(v['rendered_luma_profile_shape']+v['rendered_luma_profile_contrast']),
        chroma=8*v['chroma_pixel']+4*(v['chroma_region_mean']+v['chroma_region_residual_variation']),
        rank=10*v['luma_rank_violation'],white=v['two_pixel_region_centred_bright_shoulder'])
    assert torch.allclose(sum(groups.values()),loss,atol=1e-6)
    grads={k:torch.autograd.grad(z,nodes['up1_low_pw'],retain_graph=True)[0] for k,z in groups.items()}
    records={k:dict(loss=float(groups[k].detach()),gradient_norm=float(g.norm()),Y_gradient_norm=float(g[:,:14].norm()),C_gradient_norm=float(g[:,14:].norm())) for k,g in grads.items()}
    cos={k:{j:float((u*v).sum()/(u.norm()*v.norm()).clamp_min(1e-20)) for j,v in grads.items()} for k,u in grads.items()}
    write_json(a.out,dict(records=records,cosines=cos,total=float(loss.detach()),sum_of_terms_verified=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),readout_node='up1_low_pw',no_parameter_update=True))
    print('REFINED_GRADIENTS',records,flush=True)


if __name__=='__main__':main()
