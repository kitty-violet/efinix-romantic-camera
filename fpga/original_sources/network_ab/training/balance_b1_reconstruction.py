"""Weight-only activation range balancing, with residual-scale constraints.

Positive ReLU scale homogeneity is the idea behind cross-layer equalization.
INT8 clipping breaks unconditional equivalence, so every candidate is measured
in FP32 as well as Q17; neither equivalence nor quality is assumed.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from . import train_b1_reconstruction as r
from . import research_b1_reconstruction as s
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100561);_,plan=contract_guard()
    train,val=r.data();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=r.initialize();net.load_state_dict(parent['model']);net.quantized=False
    before=s.evaluate(net,val);original=copy.deepcopy(net.state_dict())
    maxima={};calibration=train[::max(1,len(train)//16)][:16]
    for row in calibration:
        nodes=net(r.tensor(r.read(row['path']))[None],all_nodes=True)
        for key,z in nodes.items():
            m=z.abs().amax((0,2,3))
            maxima[key]=torch.maximum(maxima.get(key,torch.zeros_like(m)),m)
    # Addition operands/result share scale. NN duplication shares its input's.
    parent_id={key:key for key in maxima}
    def find(x):
        while parent_id[x]!=x:x=parent_id[x]
        return x
    def union(x,y):
        assert maxima[x].shape==maxima[y].shape
        parent_id[find(x)]=find(y)
    for node in plan['nodes']:
        if not node['kernel']:
            for key in node['inputs']:union(node['id'],key)
    groupmax={}
    for key,m in maxima.items():
        root=find(key);groupmax[root]=torch.maximum(groupmax.get(root,torch.zeros_like(m)),m)
    trials=[]
    for target in (0,64,80,96,112):
        net.load_state_dict(original);net.quantized=False
        scales={}
        for key,m in maxima.items():
            gm=groupmax[find(key)]
            scale=(target/gm.clamp_min(1)).clamp(.25,8) if target else torch.ones_like(gm)
            scale=torch.where(gm>=125,torch.ones_like(scale),scale)
            if key in ('input','rgb_head'):scale=torch.ones_like(scale)
            scales[key]=scale
        for node in plan['nodes']:
            key=node['id'].replace('.','__')
            if key not in net.weight:continue
            gi,go=scales[node['inputs'][0]],scales[node['id']]
            if node['groups']!=1:
                net.weight[key].mul_((go/gi)[:,None,None,None])
            else:
                net.weight[key].mul_(go[:,None,None,None]/gi[None,:,None,None])
            net.bias[key].mul_(go)
        fp=s.evaluate(net,val)
        net.calibrate();state=copy.deepcopy(net.state_dict());net.quantized=True
        ptq=s.evaluate(net,val)
        eligible=fp['pooled_psnr']>=before['pooled_psnr']-.1 and fp['mean_ssim']>=before['mean_ssim']-.002
        path=out/f'target_{target}.pt'
        torch.save(dict(model=state,step=parent['step'],qat=False,contract_sha256=r.CONTRACT_SHA,
            parent=a.checkpoint,parent_sha256=file_hash(a.checkpoint),balance_target=target,
            calibrated_from_train_only=True,eligible=eligible),path)
        trials.append(dict(target=target,checkpoint=str(path),fp32=fp,ptq=ptq,eligible=eligible,
            scale_range={k:[float(v.min()),float(v.max())] for k,v in scales.items()}))
        print('BALANCE',target,'FP32',fp['pooled_psnr'],'PTQ',ptq['pooled_psnr'],'ELIGIBLE',eligible,flush=True)
        write_json(out/'results.json',dict(parent=a.checkpoint,before=before,trials=trials,
            calibration=[dict(path=z['path'],sha256=z['sha256']) for z in calibration],
            contract_sha256=r.CONTRACT_SHA,hardware_changed=False))
    best=max((t for t in trials if t['eligible']),key=lambda t:t['ptq']['pooled_psnr'])
    write_json(out/'selection.json',dict(target=best['target'],checkpoint=best['checkpoint'],
        rule='Best reviewed-validation PTQ PSNR among candidates with FP32 loss <=0.1dB and SSIM loss <=0.002; final test not read.'))


if __name__=='__main__':main()
