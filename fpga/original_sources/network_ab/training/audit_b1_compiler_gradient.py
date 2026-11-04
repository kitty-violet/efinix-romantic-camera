"""Check whether the fixed30-to15 compiler blocks final-edge supervision.

Measure gradient arriving at its clipped output versus gradient propagated to
its preactivation on real TRAIN patches. A nonzero gradient elsewhere in the
network is insufficient to establish this path is live.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_recover_boundary_gradients import recover_gradient
from .probe_b1_ordered_distance_geometry import distance_fields
from .probe_b1_decoder_coverage import fragment
from .b1_render_boundary_objective import rendering_decision_loss
from .train_b1_ternary_regions import region_codes
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


class CompilerLeakB1(OrdinalBandB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name in ('up1_low_pw','up1_high_dw'):return recover_gradient(value,pre)
        return super().training_activation_proxy(name,value,pre,r)


class CaptureCompilerB1(OrdinalBandB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name in ('up1_low_pw','up1_high_dw'):
            self.captured[name]=(value,pre)
        return super().training_activation_proxy(name,value,pre,r)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=CaptureCompilerB1;net.captured={}
    data=m.load_data(a.data);lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    records=[];leak=copy.deepcopy(net);leak.__class__=CompilerLeakB1
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2];sl=np.s_[200:520,480:800]
        x=torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        with torch.no_grad():wanted=fragment(net,F.pad(net.round(distance_fields(code,8)),(0,0,0,0,0,2)),'up1_low_dw')['up1_high_dw']
        nn=net(x,all_nodes=True);terms=rendering_decision_loss(nn['up1_high_dw'][...,20:-20,20:-20],wanted[...,20:-20,20:-20]);loss=terms[0]*16+terms[1]*8
        rec=dict(id=row['id'],nodes={})
        for name,(value,pre) in net.captured.items():
            gv,gp=torch.autograd.grad(loss,(value,pre),retain_graph=True)
            clip=(pre<0)|(pre>127);mass=gv.abs();active=mass>1e-12
            rec['nodes'][name]=dict(clipped_fraction=float(clip[:,:15].float().mean()),
                incoming_gradient_mass_blocked_fraction=float((mass*clip).sum()/mass.sum().clamp_min(1e-12)),
                incoming_active_pixels_blocked_fraction=float((active&clip).sum()/active.sum().clamp_min(1)),
                clipped_pre_gradient_max=float(gp[clip].abs().max()) if clip.any() else 0.)
        raw=net.export();fixed=leak(x,all_nodes=True)
        assert leak.export()==raw and all(torch.equal(v,fixed[k]) for k,v in nn.items())
        newloss=sum(rendering_decision_loss(fixed['up1_high_dw'][...,20:-20,20:-20],wanted[...,20:-20,20:-20]))
        g=torch.autograd.grad(newloss,fixed['up1_low_dw'])[0];assert torch.isfinite(g).all()
        records.append(rec);print('COMPILER_GRADIENT',rec,flush=True)
    write_json(a.out,dict(records=records,forward_nodes_unchanged=True,bytes_unchanged=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),training_only=True,visual_acceptance=False))


if __name__=='__main__':main()
