"""TRAIN-only optimizer diagnostic, with continuous bounded coefficients.

Use the exact same B1 graph/decoder. A differentiable training parameterization
keeps every coefficient inside its deployable INT8 row range. L-BFGS operates
on FP32 activations; each report also executes the real integer graph. There
is no inference parameterization or image postprocessing in the saved weights.
"""
import argparse
import copy
import shutil
import time
from pathlib import Path
import torch
from torch.nn.utils import parametrize
from . import train_b1_minimum_area as m
from .train_b1_ordered_regions import OrderedGradientB1
from .train_b1_count_regions import count_targets,count_loss
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .runtime import write_json,file_hash


class BoundedRows(torch.nn.Module):
    def __init__(self,limit):
        super().__init__();self.register_buffer('limit',limit)

    def forward(self,z):return self.limit*z.tanh()
    def right_inverse(self,w):return (w/self.limit).clamp(-.999999,.999999).atanh()


class BiasUnits(torch.nn.Module):
    def forward(self,z):return z*32
    def right_inverse(self,b):return b/32


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--unbalanced',action='store_true',help='Matched diagnostic: remove inverse-frequency count weights')
    ap.add_argument('--pure-only',action='store_true',help='TRAIN diagnostic: leave mixed 2x2 footprints to native coverage supervision')
    a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(712)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);state=torch.load(a.warm,map_location='cpu',weights_only=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin');net.__class__=OrderedGradientB1
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    baseline=net.export();keys=[]
    for key,w in net.weight.items():
        train=key not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
        w.requires_grad_(train);net.bias[key].requires_grad_(train)
        if train:keys.append(key)
    for key in keys:
        limit=127*(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
        parametrize.register_parametrization(net.weight,key,BoundedRows(limit))
        parametrize.register_parametrization(net.bias,key,BiasUnits())
    assert net.export()==baseline,'Parameterization must preserve initial integer coefficients exactly'
    palette=torch.tensor(state['palette']);pk=palette_key(palette);xs=[];codes=[];ids=[]
    data=m.load_data()
    for suffix,(xx,yy) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix))
        assert row['split']=='train';ids.append(row['id'])
        source=m.t.r.read(row['path'])[yy:yy+320,xx:xx+320]
        xs.append(torch.tensor(source.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        codes.append(torch.tensor(region_class_map(row['target'],row['region'],pk)[yy:yy+320,xx:xx+320].astype('int64')))
    x=torch.stack(xs);truth=[v[...,20:-20,20:-20] for v in count_targets(torch.stack(codes))]
    wanted=truth[0].argmax(1)*25+truth[1].argmax(1)*5+truth[2].argmax(1)
    pure=torch.stack([v.amax(1)>.999 for v in truth]).all(0)
    mask=pure[:,None].float() if a.pure_only else None
    parameters=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.LBFGS(parameters,lr=1,max_iter=3,max_eval=5,history_size=40,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    records=[];calls=0;started=time.perf_counter()
    def closure():
        nonlocal calls
        net.quantized=False;opt.zero_grad(set_to_none=True)
        nodes=net(x,all_nodes=True);logits=(nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
        loss=count_loss(logits,truth,mask=mask,balanced=not a.unbalanced);loss.backward();calls+=1
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        return loss
    @torch.no_grad()
    def report(step):
        rec=dict(step=step,closure_calls=calls)
        for integer in (False,True):
            net.quantized=integer;nodes=net(x,all_nodes=True)
            logits=(nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
            bits=logits>=0
            actual=bits[:,:7].sum(1)*25+bits[:,7:11].sum(1)*5+bits[:,11:].sum(1)
            label='integer' if integer else 'fp32'
            rec[label]=dict(loss=float(count_loss(logits,truth,mask=mask,balanced=not a.unbalanced)),accuracy_pure=float((actual[pure]==wanted[pure]).float().mean()))
            if integer and step==a.steps:
                for i in range(len(x)):m.t.r.c.rgb_image(nodes['rgb_head'][i]/128).save(dest/f'actual_integer_{i}.png')
        records.append(rec);print('BOUNDED_LBFGS_FIT',rec,flush=True)
        write_json(dest/'progress.json',dict(records=records,seconds=time.perf_counter()-started,fit_only=True))
    report(0)
    for step in range(1,a.steps+1):
        opt.step(closure)
        if step%10==0 or step==a.steps:report(step)
    for key in keys:
        parametrize.remove_parametrizations(net.weight,key,leave_parametrized=True)
        parametrize.remove_parametrizations(net.bias,key,leave_parametrized=True)
    net.quantized=True;blob=net.export();assert len(blob)==20480
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu')
    with torch.no_grad():
        theirs,_=actual(x*128,True);ours=net(x,all_nodes=True)
        node_error=max(float((v-theirs[k]).abs().max()) for k,v in ours.items());assert node_error==0
    result=dict(state);result.update(model=copy.deepcopy(net.state_dict()),step=a.steps,qat=False,
        optimizer_diagnostic='bounded FP32 L-BFGS on two TRAIN crops',fit_ids=ids,visual_acceptance=False,
        inference_parameterization=False,count_frequency_balanced=not a.unbalanced,pure_footprint_supervision=a.pure_only)
    torch.save(result,dest/'candidate.pt');(dest/'coefficients-padded.bin').write_bytes(blob)
    write_json(dest/'result.json',dict(warm=a.warm,warm_sha256=file_hash(a.warm),train_ids=ids,
        records=records,seconds=time.perf_counter()-started,closure_calls=calls,
        byte_equal_initial_parameterization=True,all_node_integer_error=node_error,device='cpu',
        validation_fitted=False,visual_acceptance=False,extra_inference_operations=0,
        count_frequency_balanced=not a.unbalanced,pure_footprint_supervision=a.pure_only,
        mixed_footprint_fraction=float((~pure).float().mean()),code_sha256=file_hash(__file__)))


if __name__=='__main__':main()
