"""Audit the actual current objective through every encoder activation.

Endpoint occupancy alone is not a bug: ordinal fields should saturate. Report
incoming gradient mass outside the physical clip and the actual propagated
gradient separately, along with live channel ranks and weight gradients.
"""
import argparse,types
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1,TrainableLumaReadoutTraining
from .train_b1_luma_chroma_interfaces import luminance_codes
from .b1_joint_interface_targets import joint_interface_fields
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


class CaptureReadoutB1(ReadoutB1):
    def training_activation_proxy(self,name,value,pre,r):
        out=super().training_activation_proxy(name,value,pre,r)
        if pre.requires_grad:
            out.retain_grad();pre.retain_grad();self.captured[name]=(out,pre,0 if r['activation']=='relu' else -128)
        return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=CaptureReadoutB1
    for k in net.weight:
        live=k not in ('down1__dw','down2__dw') and not k.startswith(('up1_high','up2_','rgb_head'))
        net.weight[k].requires_grad_(live);net.bias[k].requires_grad_(live)
    data=m.load_data(a.data);records=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2]
        tensor=lambda path:torch.from_numpy(m.t.r.read(path)[200:520,480:800].transpose(2,0,1).copy())[None].float()/128-1
        x=tensor(row['path']);t=tensor(row['target']);ll=torch.from_numpy(m.load_region(row['region'])[200:520,480:800].astype(np.int64))[None]
        with torch.no_grad():truth=joint_interface_fields(luminance_codes(t,ll,net.round))
        net.captured={};net.zero_grad(set_to_none=True);nodes=net(x,all_nodes=True)
        obj=object.__new__(TrainableLumaReadoutTraining);obj.net=net;obj.nodes=nodes;obj.labels=ll
        obj.class_labels=luminance_codes(t,ll,net.round);obj.motion_fraction=torch.tensor(0.);obj.args=types.SimpleNamespace(classification=2.)
        obj.distance_cache=lambda code,gain:truth
        loss,_=obj.additional_loss(nodes['rgb_head'][...,40:-40,40:-40]/128,t[...,40:-40,40:-40],False);loss.backward()
        record=dict(id=row['id'],loss=float(loss.detach()),nodes={},weights={})
        for name,(v,pre,low) in net.captured.items():
            g=v.grad;pg=pre.grad
            if g is None:continue
            outside=(pre<low)|(pre>127);mass=g.abs();vv=v.detach()[0].flatten(1).double();std=vv.std(1)
            centered=vv-vv.mean(1,keepdim=True);cov=centered@centered.T/centered.shape[1]
            eig=torch.linalg.eigvalsh(cov).clamp_min(0);prob=eig/eig.sum().clamp_min(1e-12)
            erank=float(torch.exp(-(prob*prob.clamp_min(1e-12).log()).sum()))
            record['nodes'][name]=dict(clip_fraction=float(outside.float().mean()),incoming_mass_on_clipped=float((mass*outside).sum()/mass.sum().clamp_min(1e-12)),
                pre_gradient_norm=float(pg.norm()) if pg is not None else 0.,incoming_gradient_norm=float(g.norm()),constant_channels=int((std<.05).sum()),
                channels=len(std),effective_covariance_rank=erank,channel_standard_deviations=std.tolist())
        for k,w in net.weight.items():
            if not w.requires_grad:continue
            g=w.grad
            record['weights'][k]=dict(gradient_norm=float(g.norm()) if g is not None else 0.,nonzero_gradient_rows=int((g.flatten(1).abs().sum(1)>1e-12).sum()) if g is not None else 0,
                rows=len(w),zero_weight_rows=int((w.detach().flatten(1).abs().sum(1)==0).sum()))
        records.append(record);print('READOUT_ENCODER_PATH',row['id'],{k:v['incoming_mass_on_clipped'] for k,v in record['nodes'].items() if not k.startswith('up2')},flush=True)
        write_json(a.out,dict(records=records,training_sources_only=True,complete=False,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))
    write_json(a.out,dict(records=records,training_sources_only=True,complete=True,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
