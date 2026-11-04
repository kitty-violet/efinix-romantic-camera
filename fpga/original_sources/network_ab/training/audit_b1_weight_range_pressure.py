"""Measure outward update pressure on fixed quantizer weight bounds.

Seeing a maximum code127 alone is not evidence that it blocks fitting.
Report the current centred objective's outward gradient and Adam moment.
This audit makes no changes to the model or training process.
"""
import argparse,types
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_rendered_luma_readout import RenderedLumaReadoutTraining
from .train_b1_luma_chroma_interfaces import luminance_codes
from .b1_joint_interface_targets import joint_interface_fields
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=ReadoutB1
    means={g['node_key']:g['input_mean'] for g in state['optimizer']['param_groups'] if 'node_key' in g}
    for key,w in net.weight.items():
        w.requires_grad_(key in means);net.bias[key].requires_grad_(key in means)
    data=m.load_data(a.data);records=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2];sl=np.s_[200:520,480:800]
        tensor=lambda path:torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy())[None].float()/128-1
        x=tensor(row['path']);t=tensor(row['target']);labels=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        code=luminance_codes(t,labels,net.round);truth=joint_interface_fields(code)
        obj=object.__new__(RenderedLumaReadoutTraining);obj.net=net;obj.labels=labels;obj.class_labels=code
        obj.args=types.SimpleNamespace(classification=2.);obj.motion_fraction=x.new_zeros(());obj.distance_cache=lambda code,gain:truth
        net.zero_grad();obj.nodes=net(x,all_nodes=True)
        loss,_=obj.additional_loss(obj.nodes['rgb_head'][...,40:-40,40:-40]/128,t[...,40:-40,40:-40],False);loss.backward()
        result={};total_mass=out_mass=0.
        for key,mean in means.items():
            w=net.weight[key];b=net.bias[key];s=(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
            mu=mean.reshape(-1,1,1,1) if w.shape[1]==1 and len(mean)==len(w) else mean.reshape(1,-1,1,1)
            g=w.grad-b.grad[:,None,None,None]*mu;code=net.round(w/s).clamp(-127,127)
            boundary=(code.abs()>=127);outward=boundary&(g*code<0)
            mass=float(g.detach().square().sum());out=float((g.detach().square()*outward).sum());total_mass+=mass;out_mass+=out
            result[key]=dict(boundary_count=int(boundary.sum()),outward_count=int(outward.sum()),
                outward_squared_gradient_fraction=out/max(mass,1e-20),squared_gradient=mass)
        records.append(dict(id=row['id'],nodes=result,outward_squared_gradient_fraction=out_mass/max(total_mass,1e-20)))
        print('WEIGHT_RANGE_PRESSURE',row['id'],records[-1]['outward_squared_gradient_fraction'],flush=True)
        write_json(a.out,dict(records=records,complete=False,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))
    write_json(a.out,dict(records=records,complete=True,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),training_only=True,model_changed=False))


if __name__=='__main__':main()
