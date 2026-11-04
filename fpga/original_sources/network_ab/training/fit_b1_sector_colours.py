"""Fit only 12 RGB-head coefficients to training targets; geometry unchanged."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace, write_json, file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];rows=m.load_data();training=rows['training'][::4]
    xx=torch.zeros(4,4,device='cuda',dtype=torch.float64);xy=torch.zeros(4,3,device='cuda',dtype=torch.float64);yy=0;count=0
    for i,row in enumerate(training):
        z=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)['up2_high_dw'][0,:3,::8,::8].flatten(1).T.double()/64
        x=torch.cat((z,torch.ones(len(z),1,device='cuda',dtype=torch.float64)),1)
        y=(m.t.r.tensor(m.t.r.read(row['target']))*128)[:,::8,::8].flatten(1).T.double()
        xx+=x.T@x;xy+=x.T@y;yy+=float(y.square().sum());count+=len(x)
        if (i+1)%10==0:print('SECTOR_COLOUR_SAMPLES',i+1,len(training),flush=True)
    original=torch.cat((net.weight['rgb_head'][:,:3,0,0].T.double()*64,net.bias['rgb_head'][None].double()),0)
    regularizer=torch.eye(4,device='cuda',dtype=torch.float64)*count*.0001;regularizer[-1,-1]=0
    theta=torch.linalg.solve(xx+regularizer,xy+regularizer@original)
    results=[]
    for mix in (.5,1.):
        q=original*(1-mix)+theta*mix
        net.weight['rgb_head'].zero_();net.weight['rgb_head'][:,:3,0,0]=q[:3].T.float()/64;net.bias['rgb_head'][:]=q[3].float()
        scale=net.weight['rgb_head'].abs().flatten(1).amax(1).clamp_min(1e-5)/120
        net.rgb_head_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
        sub=dest/str(mix).replace('.','p');sub.mkdir();export(net,rows['validation'],sub/'package');m.t.preview(net,rows['validation'],sub/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            manual_sector_selection=True,head_fit_only=True,ordered_regions=True,coverage_layout='15ordered360_rgb200',parent=a.warm),sub/'initial.pt')
        results.append(dict(mix=mix,rgb_head=net.weight['rgb_head'][:,:3,0,0].cpu().tolist(),bias=net.bias['rgb_head'].cpu().tolist()))
    write_json(dest/'review.json',dict(training_ids=[r['id'] for r in training],samples=count,trainable_coefficients=12,
        target_data_sha256=file_hash(m.OUT/'data.json'),parent_sha256=file_hash(a.warm),results=results,
        geometry_unchanged=True,visual_acceptance=False,added_operators=0,postfilter=False))
    shutil.copy2(__file__,dest/Path(__file__).name);print('SECTOR_COLOUR_FIT_COMPLETE',flush=True)


if __name__=='__main__':main()
