"""Measure current weighted training-gradient roles on seven TRAIN crops.

This audits the actual ternary proxy and losses. Cosines are local diagnostics,
not a proof that removing a loss improves inference. No coefficients are edited.
"""
import argparse, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_contour_coherence import ContourCoherenceTraining
from .train_b1_ternary_regions import TernaryB1,region_codes
from .probe_b1_decoder_coverage import fragment
from . import train_b1_minimum_area as m
from .runtime import file_hash,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=TernaryB1
    keys=[k for k in net.weight if k not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head','down1__dw','down2__dw')]
    weights=[net.weight[k] for k in keys]
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9));z=torch.zeros(1215,16,1,1)
    for c,(lo,hi) in enumerate(((0,7),(7,11),(11,15))):z[:,lo:hi,0,0]=(levels[:,c,None]-2*torch.arange(hi-lo)[None]).clamp(0,2)
    with torch.no_grad():palette=fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]/128
    pk=tuple(map(tuple,(palette.numpy()*128+128).tolist()));data=m.load_data(a.data);records=[]
    for video in sorted({r['video'] for r in data['training']}):
        source=[r for r in data['training'] if r['video']==video];row=source[len(source)//2]
        yy,xx=224,480;side=320;sl=np.s_[yy:yy+side,xx:xx+side]
        def read(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy())[None].float()/128-1
        x,t=read(row['path']),read(row['target']);tr=ContourCoherenceTraining.__new__(ContourCoherenceTraining)
        tr.net=net;tr.args=argparse.Namespace(classification=.5);tr.step=1;tr.rng=random.Random(4)
        tr.labels=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        tr.class_labels=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        tr.palette=palette;tr.pk=pk;tr.motion_aligned_fraction=torch.tensor(0.)
        tr.nodes=net(x,all_nodes=True);p=tr.nodes['rgb_head'][...,40:-40,40:-40]/128
        total,v=tr.additional_loss(p,t[...,40:-40,40:-40],False)
        terms={'count':.5*v['ternary_count_ce'],'flat':5.75*v['region_flatness'],
            'colour':.7*v['region_colour']+.35*v['target_pixel_colour']+1.7*v['adjacent_region_contrast'],
            'edge':1.4*v['sharp_boundary']+2*v['joint_rgb_edge_vector'],
            'tail':4*v['region_tail_flatness'],'coherence':4*v['short_contour_turn_excess'],
            'sparse':2.8*v['excess_transitions']}
        assert torch.allclose(sum(terms.values()),total,atol=1e-5)
        gradients={};layer_norms={}
        for label,value in terms.items():
            gs=torch.autograd.grad(value,weights,retain_graph=True,allow_unused=True)
            # Compare in coefficient-code units rather than arbitrarily
            # favouring rows with a tiny effective-weight quantum.
            gs=[(g if g is not None else torch.zeros_like(w))*(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None] for k,w,g in zip(keys,weights,gs)]
            gradients[label]=torch.cat([g.flatten() for g in gs]);layer_norms[label]={k:float(g.norm()) for k,g in zip(keys,gs)}
        names=list(terms);cos={i:{j:float(F.cosine_similarity(gradients[i],gradients[j],dim=0)) for j in names} for i in names}
        record=dict(id=row['id'],losses={k:float(v.detach()) for k,v in terms.items()},norms={k:float(v.norm()) for k,v in gradients.items()},cosine=cos,layer_norms=layer_norms)
        records.append(record);print('TERNARY_GRADIENT_ROLES',row['id'],record['norms'],{k:cos['count'][k] for k in ('flat','tail','coherence')},flush=True)
    write_json(a.out,dict(checkpoint_sha256=file_hash(a.checkpoint),records=records,parameters_modified=False,
        gradient='Current QAT proxy in integer coefficient-code coordinates',omitted=['temporal','phase','input stability','native-only area projection']))


if __name__=='__main__':main()
