"""Fit colour coefficients using train-only samples of the exact winner B1."""
import argparse
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .probe_b1_exclusive_winner import build
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006301);torch.set_num_threads(2)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);net=build(states=16,shape=True,soft_gate=True);d=m.load_data()
    rows=d['training'][::4];xx=torch.zeros(9,9,device='cuda',dtype=torch.float64);xy=torch.zeros(9,3,device='cuda',dtype=torch.float64)
    for i,row in enumerate(rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        nodes=net(x,all_nodes=True);features=nodes['up2_high_dw'][0,:,::4,::4].flatten(1).T.double()
        labels=torch.from_numpy(m.load_region(row['region']).astype('int64')).cuda()[None]
        boundary=torch.zeros_like(labels,dtype=torch.bool)
        boundary[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1];boundary[:,1:]|=labels[:,1:]!=labels[:,:-1]
        valid=(F.max_pool2d(boundary[:,None].float(),9,1,4)==0)[0,0,::4,::4].flatten()
        f=torch.cat((features,torch.ones_like(features[:,:1])),1)[valid];target=t[0,:,::4,::4].flatten(1).T.double()[valid]*128
        xx+=f.T@f;xy+=f.T@target
        if (i+1)%10==0:print('WINNER_COLOUR_FEATURES',i+1,len(rows),flush=True)
    current=torch.cat((net.weight['rgb_head'][:,:,0,0].T,net.bias['rgb_head'][None]),0).double()
    ridge=torch.eye(9,device='cuda',dtype=torch.float64)*xx.diag().mean()*.001;ridge[-1,-1]*=.01
    weights=torch.linalg.solve(xx+ridge,xy+ridge@current)
    net.weight['rgb_head'][:,:,0,0]=weights[:8].T;net.bias['rgb_head'][:]=weights[8]
    w=net.weight['rgb_head'];scale=w.abs().flatten(1).amax(1)/120
    net.rgb_head_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
    export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        coverage_layout='8exclusive_winner_rays',winner_shape=True,soft_gate=True,colour_fitted=True,training_steps=0),dest/'initial.pt')
    for p in (Path(__file__),Path(__file__).with_name('probe_b1_exclusive_winner.py')):shutil.copy2(p,dest/p.name)
    write_json(dest/'review.json',dict(training_ids=[r['id'] for r in rows],data_sha256=file_hash(m.OUT/'data.json'),
        weights=weights.cpu().tolist(),source_inference=True,external_filter=False,minimum_area_guaranteed=False,visual_acceptance=False))
    print('WINNER_COLOURS_COMPLETE',flush=True)


if __name__=='__main__':main()
