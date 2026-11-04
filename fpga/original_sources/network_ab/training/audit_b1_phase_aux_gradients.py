"""Measure actual auxiliary/geometry gradient contribution on TRAIN crops."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_phase_geometry_native import PhaseGeometryNativeTraining
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=ReadoutB1
    heads=nn.ModuleDict(dict(phase2=nn.Conv2d(16,4,1),phase4=nn.Conv2d(32,16,1)));heads.load_state_dict(state['heads'])
    rows={r['id']:r for r in m.load_data(a.data)['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];xs=[];ts=[];ls=[]
    for s in [mined[i] for i in (1,5,6)]:
        row=rows[s['row']['id']];x,y=s['xy'];w,h=s['wh'];sl=np.s_[y:y+h,x:x+w]
        def image(key):return torch.from_numpy(m.t.r.read(row[key])[sl].transpose(2,0,1).copy()).float()/128-1
        xs.append(image('path'));ts.append(image('target'));ls.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
    x,t,labels=map(torch.stack,(xs,ts,ls));nodes=net(x,all_nodes=True)
    obj=object.__new__(PhaseGeometryNativeTraining);obj.net=net;obj.current_nodes=nodes;obj.last_source=x;obj.heads=heads;obj.labels=labels;obj.rng=random.Random(37)
    cut=lambda v:v[...,40:-40,40:-40]
    total,v=obj.additional_loss(cut(nodes['rgb_head']/128),cut(t),False)
    groups=dict(phase_aux=2*v['source_phase_feature_aux'],coherence=4*v['short_contour_turn_excess'])
    groups['primary']=total-sum(groups.values());keys=('stem','down1__dw','down1__pw','down2__dw','down2__pw')
    parameters=[net.weight[k] for k in keys];grads={}
    for k,z in groups.items():grads[k]=torch.autograd.grad(z,parameters,retain_graph=True)
    records={}
    for i,node in enumerate(keys):
        rec={k:float(gs[i].norm()) for k,gs in grads.items()};main_grad=grads['primary'][i]
        for k in ('phase_aux','coherence'):
            rec[k+'_cosine_with_primary']=float((grads[k][i]*main_grad).sum()/(grads[k][i].norm()*main_grad.norm()).clamp_min(1e-20))
        assert all(torch.isfinite(gs[i]).all() for gs in grads.values())
        records[node]=rec
    write_json(a.out,dict(records=records,losses={k:float(z.detach()) for k,z in groups.items()},checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),no_parameter_update=True))
    print('PHASE_AUX_GRADIENTS',records,flush=True)


if __name__=='__main__':main()
