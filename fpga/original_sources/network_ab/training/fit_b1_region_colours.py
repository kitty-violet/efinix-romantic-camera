"""Fit the existing RGB head on training scenes; retain region geometry.

The probe separates colour mismatch from the accuracy of already encoded
region boundaries. It neither trains on validation images nor adds a LUT.
"""
import argparse
import shutil
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .publish_b1_style_recovery import export


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--chroma-unit',type=float,default=1.)
    ap.add_argument('--interiors',action='store_true')
    ap.add_argument('--preserve-geometry',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(20261006119);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize()
    net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
    assert state.get('coverage_layout','4Y_2Co_2Cg')=='4Y_2Co_2Cg','Fit colour-only heads before adding an outline channel'
    assert a.chroma_unit>=1
    oldunit=4.
    if not a.preserve_geometry:
        net.weight['up2_high_dw'][6:]*=oldunit/a.chroma_unit
        net.bias['up2_high_dw'][6:]*=oldunit/a.chroma_unit
        net.weight['rgb_head'][:,6:]*=a.chroma_unit/oldunit
    for key in (() if a.preserve_geometry else ('up2_high_dw','rgb_head')):
        w=net.weight[key];scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    data=m.load_data();rows=data['training'][::4] if a.interiors else data['training'][::8][:27]
    xx=torch.zeros(9,9,device='cuda',dtype=torch.float64);xy=torch.zeros(9,3,device='cuda',dtype=torch.float64)
    for row in rows:
        z=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)['up2_high_dw']
        x=z[0,:,::4,::4].flatten(1).T.double()/32;x=torch.cat((x,torch.ones_like(x[:,:1])),1)
        y=(m.t.r.tensor(m.t.r.read(row['target']))*128)[:,::4,::4].flatten(1).T.double()
        if a.interiors:
            labels=torch.from_numpy(m.load_region(row['region']).astype('int64')).cuda()
            edge=torch.zeros_like(labels,dtype=torch.bool)
            edge[1:]|=labels[1:]!=labels[:-1];edge[:,1:]|=labels[:,1:]!=labels[:,:-1]
            valid=(F.max_pool2d(edge[None,None].float(),9,1,4)[0,0,::4,::4]==0).flatten()
            x=x[valid];y=y[valid]
            assert len(x)>0
        xx+=x.T@x;xy+=x.T@y
    prior=torch.cat((net.weight['rgb_head'][:,:,0,0].T*32,net.bias['rgb_head'][None]),0).double()
    ridge=torch.eye(9,device='cuda',dtype=torch.float64)*xx.diag().mean()*1e-4;ridge[-1,-1]=0
    fit=torch.linalg.solve(xx+ridge,xy+ridge@prior).float()
    net.weight['rgb_head'].copy_(fit[:8].T[:,:,None,None]/32);net.bias['rgb_head'].copy_(fit[8])
    w=net.weight['rgb_head'];scale=(w.abs().flatten(1).amax(1).clamp_min(1e-5)/112)
    net.rgb_head_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
    export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=m.t.r.CONTRACT_SHA,
        parent=a.warm,training_head_fit=True,residual_encoding=net.package['residual_encoding'],
        field_node=state.get('field_node'),fixed_sampling=state.get('fixed_sampling',False),
        classifier_mode=state.get('classifier_mode',False)),dest/'initial.pt')
    shutil.copy2(__file__,dest/'fit_source.py')
    write_json(dest/'review.json',dict(args=vars(a),training_ids=[r['id'] for r in rows],
        parent_sha256=file_hash(a.warm),code_sha256=file_hash(__file__),weights=fit.tolist(),
        operators_added=0,visual_accepted=False))
    print('REGION_HEAD_FIT',dest,flush=True)


if __name__=='__main__':main()
