"""CPU train-only linear head fit: diagnose optimization vs frozen features.

Fits the existing fifteen affine rows analytically on representative TRAIN
crops. No target injection at inference, additional operator, or hidden filter.
This is a diagnostic initializer; images do not imply visual acceptance.
"""
import argparse
from pathlib import Path
import copy
import json
import cv2
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .probe_b1_independent_decoder import native_distances
from .build_b1_ordered_palette import bits
from .build_b1_binary_palette import calibrate_rows
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();configure_workspace();torch.set_num_threads(2);cv2.setNumThreads(2)
    dest=Path(a.out);dest.mkdir(exist_ok=False);s=torch.load(a.warm,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(s['model']);net.package['residual_encoding']=s['residual_encoding'];net.quantized=True
    table=bits('cpu');pk=palette_key(torch.tensor(s['palette']));data=m.load_data()
    rng=np.random.default_rng(20261006153);xs=[];ys=[];samples=[]
    for n,row in enumerate(data['training'][::4]):
        assert row['split']=='train'
        source=m.t.r.read(row['path']);labels=region_class_map(row['target'],row['region'],pk)
        for k in range(2):
            x=int(rng.integers(0,961));y=int(rng.integers(0,401))
            patch=source[y:y+320,x:x+320];lab=labels[y:y+320,x:x+320]
            inp=torch.tensor(patch.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
            f=net(inp,all_nodes=True)['up1_low_dw'][0,:,10:-10:2,10:-10:2]
            target=native_distances(table[torch.tensor(lab.astype(np.int64))].permute(2,0,1)[None])[0,:,10:-10:2,10:-10:2]
            xs.append(f.flatten(1).T.double());ys.append(target.flatten(1).T.double())
            samples.append(dict(id=row['id'],xy=[x,y],wh=[320,320],split='train'))
        if (n+1)%10==0:print('DISTANCE_HEAD_FEATURES',n+1,flush=True)
    x=torch.cat(xs);y=torch.cat(ys);mu=x.mean(0);sd=x.std(0).clamp_min(2)
    design=torch.cat(((x-mu)/sd,torch.ones(len(x),1,dtype=torch.float64)),1)
    fitted=[];old=[];after=[]
    for i in range(15):
        positive=(y[:,i]>=0).double();rate=positive.mean().clamp(.125,.875)
        weight=positive*.5/rate+(1-positive)*.5/(1-rate)
        # Equalize sparse classes; keep a modest preference for boundary bands.
        weight=weight*(1+(y[:,i].abs()<2).double())
        ridge=torch.eye(33,dtype=torch.float64)*weight.sum()*.002;ridge[-1,-1]=0
        theta=torch.linalg.solve(design.T@(design*weight[:,None])+ridge,
            design.T@(y[:,i]*weight))
        fitted.append(theta)
        q=(net.up1_low_pw_m/131072).flatten()[i]
        ow=torch.round(net.weight['up1_low_pw'][i,:,0,0]/q).clamp(-127,127)*q
        ob=torch.round(net.bias['up1_low_pw'][i]/q)*q
        old.append(float((((x@ow.double()+ob-63.5)/8-y[:,i]).abs()*weight).sum()/weight.sum()))
        after.append(float(((design@theta-y[:,i]).abs()*weight).sum()/weight.sum()))
    theta=torch.stack(fitted,1)
    w=(theta[:32].T/sd)*8;b=63.5+theta[-1]*8-w@mu
    net.weight['up1_low_pw'][:15,:,0,0]=w.float();net.bias['up1_low_pw'][:15]=b.float()
    calibrate_rows(net,['up1_low_pw'])
    # Check the quantized rows too, not just the analytic FP64 solution.
    qw=[];qb=[]
    for i in range(15):
        q=(net.up1_low_pw_m/131072).flatten()[i]
        qw.append((torch.round(net.weight['up1_low_pw'][i,:,0,0]/q).clamp(-127,127)*q).double())
        qb.append((torch.round(net.bias['up1_low_pw'][i]/q)*q).double())
    quant_error=((x@torch.stack(qw).T+torch.stack(qb)-63.5)/8-y).abs().mean(0).tolist()
    changed={k:v for k,v in net.state_dict().items() if not torch.equal(v,s['model'][k])}
    assert set(changed)<= {'weight.up1_low_pw','bias.up1_low_pw','up1_low_pw_m'}
    result=copy.deepcopy(s);result.update(model=net.state_dict(),step=0,distance_head_closed_form=True,
        parent=a.warm,diagnostic_initializer=True,minimum_area_inference_guarantee=False)
    # This is a new initializer, not an optimizer/RNG resume point.
    for key in ('optimizer','rng','random_state','torch_rng','cuda_rng'):result.pop(key,None)
    torch.save(result,dest/'initial.pt');net.export()
    records=[]
    for row in data['validation']:
        src=m.t.r.read(row['path']);inp=torch.tensor(src.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        p=net(inp)[0][0];m.t.r.c.rgb_image(p).save(dest/(row['id']+'_actual_B1.png'))
        truth=torch.tensor(m.t.r.read(row['target']).transpose(2,0,1).copy(),dtype=torch.float32)/128-1
        records.append(dict(id=row['id'],original_target_mae=float((p-truth).abs().mean()*128)))
    write_json(dest/'review.json',dict(parent_sha256=file_hash(a.warm),training_samples=samples,
        scalar_training_samples=len(x),old_weighted_distance_mae=old,fit_weighted_distance_mae=after,
        quant_unweighted_distance_mae=quant_error,validation=records,changed_state_keys=list(changed),
        graph_changed=False,postfilter=False,visual_accepted=False,device='cpu'))
    print('DISTANCE_HEAD_REFIT_COMPLETE',dict(old_mean=float(np.mean(old)),fit_mean=float(np.mean(after)),validation=records),flush=True)


if __name__=='__main__':main()
