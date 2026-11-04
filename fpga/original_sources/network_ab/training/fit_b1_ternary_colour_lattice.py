"""Fit the existing RGB lattice to TRAIN region colours, not model errors.

Unlike the previous RGB correction on erroneous predicted fields, alternate
nearest *constant* representable codes and a constrained existing-head fit.
All assignments are training-only. Exact INT8/Q17 palettes score each proposal.
No runtime LUT or additional coefficients are introduced.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .probe_b1_ternary_field_targets import canonical_states
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash


@torch.inference_mode()
def constants(net):
    z=F.pad(canonical_states(torch.arange(1215)[:,None,None]),(0,0,0,0,0,1))
    nodes=fragment(net,z,'up2_low_dw')
    return nodes['up2_high_dw'][:,:3,0,0],nodes['rgb_head'][:,:,0,0]


@torch.inference_mode()
def assign(target,pal,weight):
    indices=[]
    for part in target.split(2048):
        dist=part.square().sum(1)[:,None]+pal.square().sum(1)[None]-2*part@pal.T
        indices.append(dist.argmin(1))
    ids=torch.cat(indices);delta=pal[ids]-target
    return ids,float((delta.square().mean(1)*weight).sum()),float((delta.abs().mean(1)*weight).sum())


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    rows=m.load_data(args.data)['training'];source_count={}
    for row in rows:source_count[row['video']]=source_count.get(row['video'],0)+1
    targets=[];weights=[]
    for row in rows:
        rgb=m.t.r.read(row['target']).astype(np.float64);labels=m.load_region(row['region']).astype(np.int64)
        count=np.bincount(labels.ravel());keep=count>0
        mean=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel(),minlength=len(count))/count.clip(1) for c in range(3)],1)
        w=np.sqrt(count[keep]);w/=w.sum()*source_count[row['video']]
        targets.append(mean[keep]-128);weights.append(w)
    target=torch.from_numpy(np.concatenate(targets)).float();weight=torch.from_numpy(np.concatenate(weights)).float();weight/=weight.sum()
    feature,pal=constants(net);design=torch.cat((feature,torch.ones(len(feature),1)),1).double()
    best=float('inf');best_state=None;records=[];prior=None
    for step in range(16):
        feature,pal=constants(net);ids,mse,mae=assign(target,pal,weight)
        records.append(dict(step=step,mse=mse,mae=mae,unique_rgb=int(torch.unique(pal,dim=0).shape[0])))
        print('TERNARY_COLOUR_FIT',records[-1],flush=True)
        if mse<best:best=mse;best_state=copy.deepcopy(net.state_dict());best_step=step
        x=design[ids];ww=weight.double();y=target.double()
        gram=x.T@(x*ww[:,None]);cross=x.T@(y*ww[:,None])
        if prior is None:prior=torch.cat((net.weight['rgb_head'][:,:3,0,0].T,net.bias['rgb_head'][None]),0).double()
        ridge=gram.diag().clamp_min(.01)*.003
        solution=torch.linalg.solve(gram+torch.diag(ridge),cross+ridge[:,None]*prior)
        # Restrict this diagnostic to a local calibration, no large hue rotation
        # or near-singular matrix accepted merely for lower average colour loss.
        delta=(solution-prior).clamp(-1.5,1.5);delta[3]=(solution-prior)[3].clamp(-12,12)
        solution=prior+delta
        net.weight['rgb_head'].zero_();net.weight['rgb_head'][:,:3,0,0]=solution[:3].T.float();net.bias['rgb_head'].copy_(solution[3].float())
        mult=(net.weight['rgb_head'].abs().flatten(1).amax(1)*131072/112).ceil().clamp(1,131071)
        net.rgb_head_m.copy_(mult.double()[None,:,None,None])
    net.load_state_dict(best_state);raw=net.export();assert len(raw)==20480
    (dest/'coefficients-padded.bin').write_bytes(raw)
    saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng','bounded_training_model')}
    saved.update(model=copy.deepcopy(net.state_dict()),colour_lattice_fit=dict(training_only=True,selected=best_step),
        parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),visual_acceptance=False)
    torch.save(saved,dest/'initial.pt')
    write_json(dest/'review.json',dict(training_regions=len(target),source_videos=len(source_count),records=records,
        selected_step=best_step,sha256=hashlib.sha256(raw).hexdigest(),target_labels_not_predictions=True,
        extra_inference_operations=0,visual_acceptance=False))


if __name__=='__main__':main()
