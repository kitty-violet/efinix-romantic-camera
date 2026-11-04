"""Reuse inactive existing ReLU channels, preserving the starting function.

No added operators/parameters. Calibration uses training images only.
"""
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_style_recovery as t
from .runtime import write_json,file_hash,configure_workspace
from .engine import set_reproducible


@torch.no_grad()
def main():
    configure_workspace();set_reproducible(2026100683)
    dest=t.OUT/'revived-initialization';dest.mkdir(exist_ok=True)
    assert not (dest/'initial.pt').exists(), 'Do not overwrite a verified initialization'
    parent=t.OUT/'structured/best.pt';state=torch.load(parent,map_location='cpu',weights_only=True)
    net=t.r.initialize();net.load_state_dict(state['model']);net.quantized=False
    rows=t.data()['training'];samples=[]
    for i,row in enumerate(rows[::max(1,len(rows)//24)][:24]):
        a=t.r.read(row['path']);y=(i*73)%464;x=(i*137)%1024
        samples.append(t.r.tensor(a[y:y+256,x:x+256])[None])
    original=[net(x)[0].clone() for x in samples]
    chains=[('res1_conv1.pw',['res1_conv2.dw'],'res1_conv2.pw'),
            ('res2_conv1.pw',['res2_conv2.dw'],'res2_conv2.pw'),
            ('res3_conv1.pw',['res3_conv2.dw'],'res3_conv2.pw'),
            ('up1_low_pw',['up1_high_dw','up2_low_dw'],'up2_low_pw'),
            ('up2_low_pw',['up2_high_dw'],'rgb_head')]
    changes=[]
    for name,depths,consumer in chains:
        key=name.replace('.','__');ck=consumer.replace('.','__')
        values=[];maximum=[]
        for x in samples:
            a=net(x,all_nodes=True)[name].flatten(2)[0]
            values.append(a[:,::17].cpu());maximum.append(a.abs().amax(1).cpu())
        values=torch.cat(values,1)
        dead=torch.where(torch.stack(maximum).amax(0)<1e-7)[0].tolist()
        live=torch.argsort(values.std(1),descending=True).tolist()
        for j,ch in enumerate(dead):
            # An all-zero PW row produces only constants through its depthwise
            # successors. Fold those constants before connecting a new hinge.
            const=0.
            for dw in depths:
                dk=dw.replace('.','__')
                const=float((net.weight[dk][ch].sum()*const+net.bias[dk][ch]).clamp(0,127))
            net.bias[ck].add_(net.weight[ck][:,ch,0,0]*const)
            net.weight[ck][:,ch].zero_()
            source=live[j%len(live)];q=(.25,.5,.75)[j%3]
            threshold=float(torch.quantile(values[source],q))
            net.weight[key][ch].copy_(net.weight[key][source])
            net.bias[key][ch].copy_(net.bias[key][source]-threshold)
            for dw in depths:
                dk=dw.replace('.','__');net.weight[dk][ch].zero_()
                net.weight[dk][ch,0,1,1]=1;net.bias[dk][ch]=0
            changes.append(dict(node=name,channel=ch,source_channel=source,
                                threshold=threshold,constant_folded=const,consumer=consumer))
    errors=[float((net(x)[0]-old).abs().max()) for x,old in zip(samples,original)]
    assert max(errors)<1e-4,errors
    state['model']=copy.deepcopy(net.state_dict());state['step']=0
    state['qat']=False;state['args']['qat']=False
    torch.save(state,dest/'initial.pt')
    # Check new rows do contribute a real gradient through the existing graph.
    with torch.enable_grad():
        x=samples[0];p=net(x)[0];goal=torch.zeros_like(p)
        F.mse_loss(p,goal).backward()
        gradient={v['consumer']:float(net.weight[v['consumer'].replace('.','__')].grad.abs().max()) for v in changes}
    assert changes and all(v>0 for v in gradient.values())
    write_json(dest/'review.json',dict(parent=str(parent),parent_sha256=file_hash(parent),
        calibration_source='24 training-only crops',changed=changes,
        max_initial_output_error=max(errors),consumer_gradient=gradient,
        hardware_changed=False,extra_parameters=0))
    print(json.dumps(dict(revived=len(changes),max_error=max(errors),changes=changes)),flush=True)


if __name__=='__main__':main()
