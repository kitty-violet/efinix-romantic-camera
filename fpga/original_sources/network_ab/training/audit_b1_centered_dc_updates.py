"""Measure actual quantized affine DC jumps under two optimizers on TRAIN crops."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from .fit_b1_distance_geometry import optimizer
from .centered_b1_optimizer import CenteredB1Adam,calibration_means
from .probe_b1_encoder_gauge import load
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def actual_parameters(net):
    result={}
    for k,w in net.weight.items():
        if not w.requires_grad:continue
        s=(getattr(net,k+'_m')/131072).float().flatten()
        result[k]=(net.round(w/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None],net.round(net.bias[k]/s)*s)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);base.__class__=OrdinalBandB1;data=m.load_data(a.data);dest=Path(a.out)
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720'];pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    videos=sorted({r['video'] for r in data['training']});xs=[];codes=[];ids=[]
    for v in (videos[0],videos[len(videos)//2],videos[-1]):
        rows=[r for r in data['training'] if r['video']==v];r=rows[len(rows)//2];ids.append(r['id']);sl=np.s_[280:440,560:720]
        xs.append(torch.from_numpy(m.t.r.read(r['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
        codes.append(torch.from_numpy(region_codes(r['target'],r['region'],pk)[sl].astype(np.int64)))
    x=torch.stack(xs);truth=distance_fields(torch.stack(codes),8);means=calibration_means(base,x);records=[]
    for mode in ('raw','centred'):
        net=copy.deepcopy(base);opt=optimizer(net,centred=mode=='centred',means=means);params=[p for p in net.parameters() if p.requires_grad]
        for step in range(16):
            nodes=net(x,all_nodes=True);full,band=distance_objective(nodes['up1_low_dw'][...,10:-10,10:-10],truth[...,10:-10,10:-10]);loss=16*full+8*band
            opt.zero_grad();loss.backward()
            if isinstance(opt,CenteredB1Adam):opt.prepare_gradients()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);before=actual_parameters(net);opt.step();after=actual_parameters(net)
            for k,(w,b) in before.items():
                ww,bb=after[k];mu=means[k];mu=mu[:,None,None,None] if w.shape[1]==1 and len(mu)==len(w) else mu[None,:,None,None]
                movement=((ww-w)*mu).sum((1,2,3));dc=movement+bb-b
                records.append(dict(mode=mode,step=step+1,node=k,weight_code_changes=int((ww!=w).sum()),weight_dc_max=float(movement.abs().max()),bias_change_max=float((bb-b).abs().max()),affine_dc_max=float(dc.abs().max())))
        print('CENTERED_DC_AUDIT',mode,'max affine jump',max(r['affine_dc_max'] for r in records if r['mode']==mode),flush=True)
    write_json(dest,dict(records=records,train_ids=ids,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),scope='16 local QAT updates; affine response at fixed calibrated mean, not video flicker or visual acceptance'))


if __name__=='__main__':main()
