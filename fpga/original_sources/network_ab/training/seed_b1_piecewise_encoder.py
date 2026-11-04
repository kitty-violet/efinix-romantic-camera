"""Activate spatially selective residual features in the existing B1 encoder.

The reconstruction-derived encoder leaves most PW ReLUs in their linear part.
For each existing residual block, place existing PW biases at TRAIN quantiles,
then absorb the constant offset into the following DW bias. Above the new
threshold the affine computation is preserved; below it the feature is floored.
This is deliberately NOT an equivalent rewrite. It initializes nonlinear
features for subsequent training, and must be validated on actual output.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import preview,evaluate
from .runtime import write_json,file_hash,configure_workspace


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();configure_workspace();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
    data=m.load_data(args.data);rows=[];seen=set()
    for row in data['training']:
        if row['video'] not in seen:seen.add(row['video']);rows.append(row)
    report=[]
    for block in (1,2,3):
        name=f'res{block}_conv1.pw';key=name.replace('.','__');nextkey=f'res{block}_conv2__dw'
        features=[]
        for row in rows:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None]
            z=net(x,all_nodes=True)[name][0,:,::4,::4].flatten(1);features.append(z)
        a=torch.cat(features,1)
        # Staggered quantiles avoid making every row the same threshold.
        quantiles=torch.linspace(.2,.8,32,device=a.device).roll((block-1)*9)
        thresholds=torch.stack([torch.quantile(v,q) for v,q in zip(a,quantiles)]).floor()
        spread=a.std(1);thresholds=torch.where((spread>2)&(thresholds>2)&(thresholds<120),thresholds,0.)
        net.bias[key].sub_(thresholds)
        net.bias[nextkey].add_(thresholds*net.weight[nextkey].sum((1,2,3)))
        report.append(dict(node=name,thresholds=thresholds.tolist(),spread=spread.tolist(),
            active_rows=int((thresholds>0).sum()),quantiles=quantiles.tolist()))
    saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng','bounded_training_model')}
    saved.update(model=copy.deepcopy(net.state_dict()),piecewise_encoder_initialization=report,
        parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),
        added_training_steps=0,manual_bias_initialization=True,visual_acceptance=False,
        added_inference_operators=0,source_to_style=True)
    torch.save(saved,dest/'initial.pt');raw=net.export();(dest/'coefficients-padded.bin').write_bytes(raw)
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda');errors=[]
    for row in data['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];ours=net(x,all_nodes=True);ref,_=actual(x*128,True)
        errors.append(max(float((v-ref[k]).abs().max()) for k,v in ours.items()))
    assert max(errors)==0
    preview(net,data['validation'],dest/'preview')
    write_json(dest/'review.json',dict(parent=args.checkpoint,parent_sha256=file_hash(args.checkpoint),
        training_ids=[r['id'] for r in rows],blocks=report,all_node_error=max(errors),bytes=len(raw),
        validation=evaluate(net,data['validation']),not_equivalent_rewrite=True,visual_acceptance=False))
    print('PIECEWISE_ENCODER_INITIALIZED',[(v['node'],v['active_rows']) for v in report],max(errors),flush=True)


if __name__=='__main__':main()
