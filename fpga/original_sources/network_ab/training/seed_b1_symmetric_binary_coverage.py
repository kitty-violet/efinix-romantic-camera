"""Fit only the new binary-score readout, using21 TRAIN frames.

The fixed weighted binary coverage decoder was tested separately. Validation
frames are evaluated after fitting and never enter the ridge/likelihood fit.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .b1_binary_layout import fields_for_layout
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import preview,evaluate
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash,configure_workspace


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);torch.manual_seed(723);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);net=net.cuda();setup=state['binary_coverage720'];data=m.load_data(a.data)
    levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda')).float()
    pal=(levels@levels.new_tensor(setup['matrix']).T+levels.new_tensor(setup['bias'])+128).clamp(0,255);pk=tuple(map(tuple,pal.cpu().tolist()))
    grouped={}
    for row in data['training']:grouped.setdefault(row['video'],[]).append(row)
    chosen=[]
    for group in grouped.values():
        group=sorted(group,key=lambda r:r['id']);chosen.extend(group[i] for i in sorted(set((0,len(group)//2,len(group)-1))))
    xx=[];yy=[]
    with torch.no_grad():
        for row in chosen:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];nodes=net(x,all_nodes=True);feature=nodes['up1_low_dw'][0,:,::3,::3].flatten(1).T
            code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None].cuda()
            truth=F.avg_pool2d(fields_for_layout(code,setup),4)[0,:,::3,::3].flatten(1).T
            xx.append(feature);yy.append(truth)
        x=torch.cat(xx);truth=torch.cat(yy);mu=x.mean(0);sd=x.std(0).clamp_min(1);xn=(x-mu)/sd
        design=torch.cat((xn,torch.ones(len(x),1,device='cuda')),1)
        penalty=torch.eye(33,device='cuda')*.01;penalty[-1,-1]=0
        theta=torch.linalg.solve(design.T@design/len(x)+penalty,design.T@((truth-.5)*4)/len(x))
    theta=torch.nn.Parameter(theta);opt=torch.optim.Adam([theta],lr=.01);history=[]
    for i in range(400):
        logits=design@theta;loss=F.binary_cross_entropy_with_logits(logits,truth)+.0001*theta[:-1].square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (i+1)%100==0:
            history.append(dict(step=i+1,loss=float(loss.detach())));print('SYMMETRIC_READOUT_FIT',history[-1],flush=True)
    with torch.no_grad():
        # 12 score units per logit leaves ample room around centre64 while
        # retaining a useful confidence margin for the fixed clipping ramp.
        w=(theta[:-1].T/sd[None])*12;b=theta[-1]*12+64-w@mu
        key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_();net.weight[key][:15,:,0,0]=w;net.bias[key][:15]=b
        mult=(net.weight[key].abs().flatten(1).amax(1).clamp_min(1e-5)*131072/120).ceil().clamp(1,131071)
        getattr(net,key+'_m').copy_(mult.double()[None,:,None,None]);raw=net.export();assert len(raw)==20480
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')
        for row in data['validation']:
            src=m.t.r.tensor(m.t.r.read(row['path']))[None];nodes=net(src,all_nodes=True);other,_=actual(src*128,True)
            assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        saved=copy.deepcopy(state);setup=copy.deepcopy(setup);setup['source_readout_not_fitted']=False
        saved.update(model={k:v.cpu() for k,v in net.state_dict().items()},binary_coverage720=setup,
            source_readout_not_fitted=False,readout_train_ids=[r['id'] for r in chosen],step=0,visual_acceptance=False)
        torch.save(saved,dest/'initial.pt');(dest/'initial.bin').write_bytes(raw)
        preview(net,data['validation'],dest/'preview');review=evaluate(net,data['validation'])
    write_json(dest/'review.json',dict(training_ids=[r['id'] for r in chosen],training_sources=len(grouped),history=history,
        validation=review,all_six720_node_error=0,byte_size=20480,visual_acceptance=False,
        template_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data)))
    print('SYMMETRIC_READOUT_COMPLETE',review['mean'],flush=True)


if __name__=='__main__':main()
