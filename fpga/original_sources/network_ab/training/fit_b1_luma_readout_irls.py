"""TRAIN-only robust linear calibration of existing14 Y readout rows.

This isolates the information already present in32 latent fields. A small
regularized censored least-squares solve does not add a runtime model. Each
candidate is quantized into the unchanged B1 and actually replayed. No claim
of temporal quality or global capacity follows from this controlled fit.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image
from .probe_b1_encoder_gauge import load
from .train_b1_luma_chroma_interfaces import luminance_codes
from .train_b1_trainable_luma_readout import backend_palette
from .b1_distance_target_cache import DistanceTargetCache
from .b1_joint_interface_targets import joint_interface_fields
from .probe_b1_monotone_pair_decoder import make_board
from .audit_b1_rim_tolerance import rim_terms
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);assert torch.all(base.up1_low_pw_m==2048)
    data=m.load_data(a.data);out=Path(a.out);out.mkdir(exist_ok=False);chosen=[]
    for vid in sorted({r['video'] for r in data['training']}):
        rows=sorted([r for r in data['training'] if r['video']==vid],key=lambda r:r['id'])
        chosen.extend(rows[i] for i in sorted({0,len(rows)//2,len(rows)-1}))
    tensor=lambda path:torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
    cache=DistanceTargetCache('E:/fpga/model-training-workspace/cache/b1-joint-interface-fp32',compute=joint_interface_fields)
    xs=[];ys=[]
    for row in chosen:
        x=tensor(row['path']);t=tensor(row['target']);lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        z=base(x,all_nodes=True)['up1_low_dw'];truth=cache(luminance_codes(t,lab,base.round),8)[:,:14]
        # Deterministic sublattice covers the full native frame, not one ROI.
        phase=len(xs)%4;xs.append(z[0,:,phase::4,phase::4].flatten(1).T.double()/127)
        ys.append(truth[0,:,phase::4,phase::4].flatten(1).T.double()/127)
        print('READOUT_IRLS_SAMPLE',row['id'],flush=True)
    x=torch.cat(xs);y=torch.cat(ys);fit_samples=len(x);mu=x.mean(0);sigma=x.std(0).clamp_min(.05)
    X=torch.cat(((x-mu)/sigma,torch.ones(len(x),1,dtype=torch.float64)),1)
    W0=base.weight['up1_low_pw'][:14,:,0,0].double();b0=base.bias['up1_low_pw'][:14].double()/127
    prior=torch.cat(((W0*sigma).T,(W0@mu+b0)[None]),0);theta=prior.clone()
    prior_penalty=torch.eye(33,dtype=torch.float64)*.001;prior_penalty[-1,-1]=1e-6
    steps=[]
    for step in range(8):
        pred=X@theta;residual=pred-y
        valid=~(((y<=0)&(pred<=0))|((y>=1)&(pred>=1)))
        weight=(.08/residual.abs().clamp_min(.08))*valid*(1+2*((y>0)&(y<1)))
        columns=[]
        for j in range(14):
            w=weight[:,j];den=w.sum().clamp_min(1);cov=(X.T*w)@X/den
            rhs=X.T@(w*(y[:,j]-X@prior[:,j]))/den
            columns.append(prior[:,j]+torch.linalg.solve(cov+prior_penalty,rhs))
        theta=.5*theta+.5*torch.stack(columns,1)
        err=float((X@theta).clamp(0,1).sub(y).abs().mean())
        steps.append(dict(step=step+1,training_field_mae127=err*127));print('READOUT_IRLS',steps[-1],flush=True)
    W=(theta[:32]/sigma[:,None]).T;b=(theta[32]-W@mu)*127
    assert torch.isfinite(W).all() and torch.isfinite(b).all()
    net=copy.deepcopy(base);s=1/64
    Wq=net.round(W.float()/s).clamp(-127,127)*s
    # Correct the intercept for the actual coefficient rounding on TRAIN mean.
    bq=b.float()+((W.float()-Wq)*mu.float()[None]).sum(1)*127
    net.weight['up1_low_pw'][:14,:,0,0].copy_(Wq);net.bias['up1_low_pw'][:14].copy_(net.round(bq/s)*s)
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    assert torch.equal(backend_palette(base),backend_palette(net));boards=[{}, {}, {}];records=[]
    for row in data['validation']:
        x=tensor(row['path']);t=tensor(row['target']);lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        before=base(x,all_nodes=True);after=net(x,all_nodes=True);nodes,_=actual(net.round(x*128),True)
        assert all(torch.equal(v,nodes[k]) for k,v in after.items())
        assert torch.equal(before['up1_low_pw'][:,14:],after['up1_low_pw'][:,14:])
        boards[0][row['id']]=Image.fromarray(m.t.r.read(row['target']))
        for j,(mode,nn) in enumerate((('before',before),('after',after)),1):
            p=nn['rgb_head']/128;rim,_=rim_terms(p,t,lab,2);_,terms=reference_boundary_profile(p,t,lab)
            record=dict(id=row['id'],mode=mode,mae=float((p-t).abs().mean()*128),tight_rim=float(rim),
                profile_shape=float(terms['rgb_profile_shape']),profile_contrast=float(terms['rgb_profile_contrast']),
                rank_violation_fraction=float((torch.diff(nn['up1_low_pw'][:,:14],dim=1)>0).float().mean()))
            records.append(record);im=m.t.r.c.rgb_image(p[0]);boards[j][row['id']]=im;im.save(out/f'{mode}_{row["id"]}.png')
        print('READOUT_IRLS_VALIDATION',records[-2:],flush=True)
    make_board(boards,['目标','校准前','仅Y读出稳健拟合：未验收'],out/'comparison.png')
    saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
    saved.update(model=copy.deepcopy(net.state_dict()),trainable_luma_readout=True,luma_supervision_node='up1_low_pw',
        encoder32_channels_are_latent=True,readout_irls_calibration=True,calibration_ids=[r['id'] for r in chosen],visual_acceptance=False)
    torch.save(saved,out/'candidate.pt');write_json(out/'review.json',dict(records=records,steps=steps,calibration_ids=[r['id'] for r in chosen],
        native_training_frames=len(chosen),fit_samples=fit_samples,backend_constant1215_error=0,all_nodes_integer_error=0,chroma_fields_unchanged=True,
        bytes=20480,extra_inference_operations=0,visual_acceptance=False,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__)))


if __name__=='__main__':main()
