"""Measure three weight-only equalizations against STYLE, not source fidelity."""
import argparse
import copy
import torch
from pathlib import Path
from . import train_b1_style_recovery as m
from . import train_b1_reconstruction as r
from .engine import set_reproducible
from .runtime import configure_workspace,contract_guard,file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100586);_,plan=contract_guard();d=m.data()
    dest=Path(a.out);dest.mkdir(exist_ok=False)
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=r.initialize();net.load_state_dict(parent['model']);net.quantized=False
    original=copy.deepcopy(net.state_dict());before=m.evaluate(net,d['validation']);maxima={}
    calibration=d['training'][::max(1,len(d['training'])//16)][:16]
    for row in calibration:
        for key,z in net(r.tensor(r.read(row['path']))[None],all_nodes=True).items():
            value=z.abs().amax((0,2,3));maxima[key]=torch.maximum(maxima.get(key,torch.zeros_like(value)),value)
    parents={k:k for k in maxima}
    def find(x):
        while parents[x]!=x:x=parents[x]
        return x
    for row in plan['nodes']:
        if not row['kernel']:
            for key in row['inputs']:parents[find(row['id'])]=find(key)
    groupmax={}
    for key,value in maxima.items():
        root=find(key);groupmax[root]=torch.maximum(groupmax.get(root,torch.zeros_like(value)),value)
    trials=[]
    for target in (0,80,112):
        net.load_state_dict(original);net.quantized=False;scales={}
        for key in maxima:
            value=groupmax[find(key)]
            scale=(target/value.clamp_min(1)).clamp(.25,8) if target else torch.ones_like(value)
            scale=torch.where(value>=125,torch.ones_like(scale),scale)
            if key in ('input','rgb_head'):scale=torch.ones_like(scale)
            scales[key]=scale
        for row in plan['nodes']:
            key=row['id'].replace('.','__')
            if key not in net.weight:continue
            gi,go=scales[row['inputs'][0]],scales[row['id']]
            net.weight[key].mul_((go/gi)[:,None,None,None] if row['groups']!=1 else go[:,None,None,None]/gi[None,:,None,None])
            net.bias[key].mul_(go)
        fp=m.evaluate(net,d['validation']);net.calibrate();net.quantized=True
        before_bias=m.evaluate(net,d['validation']);best=before_bias['mean']['score'];chosen=copy.deepcopy(net.state_dict())
        # Sequential activation-mean correction fitted on training sources only.
        xs=[r.tensor(r.read(z['path']))[None] for z in calibration[::3][:6]];wanted={}
        net.quantized=False
        for x in xs:
            nodes=net(x,all_nodes=True)
            for row in plan['nodes']:
                if row['kernel']:
                    key=row['id'];value=nodes[key].mean((0,2,3));wanted[key]=wanted.get(key,torch.zeros_like(value))+value/len(xs)
        net.quantized=True
        for row in plan['nodes']:
            if not row['kernel']:continue
            mean=torch.zeros_like(wanted[row['id']]);active=torch.zeros_like(mean)
            for x in xs:
                z=net(x,all_nodes=True)[row['id']];mean+=z.mean((0,2,3))/len(xs)
                ok=(z>0)&(z<127) if row['activation']=='relu' else (z>-128)&(z<127)
                active+=ok.float().mean((0,2,3))/len(xs)
            delta=((wanted[row['id']]-mean)/active.clamp_min(.2)).clamp(-8,8)
            net.bias[row['id'].replace('.','__')].add_(delta)
        corrected=m.evaluate(net,d['validation']);use_bias=corrected['mean']['score']<best
        if not use_bias:net.load_state_dict(chosen)
        chosen_eval=corrected if use_bias else before_bias
        eligible=fp['mean']['score']<=before['mean']['score']*1.02
        path=dest/f'target_{target}.pt'
        torch.save(dict(model=net.state_dict(),step=parent['step'],qat=True,contract_sha256=r.CONTRACT_SHA,
            parent=a.checkpoint,parent_sha256=file_hash(a.checkpoint),balance_target=target,bias_correction=use_bias),path)
        trials.append(dict(target=target,fp32=fp,ptq=before_bias,corrected=corrected,chosen=chosen_eval,
            eligible=eligible,checkpoint=str(path),bias_correction_selected=use_bias))
        write_json(dest/'results.json',dict(before=before,trials=trials,calibration=calibration,test_used=False))
        print('STYLE_CALIBRATION',target,chosen_eval['mean']['score'],use_bias,flush=True)
    best=min((x for x in trials if x['eligible']),key=lambda x:x['chosen']['mean']['score'])
    write_json(dest/'selection.json',dict(checkpoint=best['checkpoint'],target=best['target'],
        rule='Style validation score; exclude >2% floating degradation; reserved test not used'))


if __name__=='__main__':main()
