"""Activate unused nonlinear regimes while preserving the learned decoder.

Shift selected residual hidden biases towards measured training quantiles;
compensate the following DW bias for the removed constant. Above threshold the
branch function is unchanged, below threshold it gains a trainable hinge.
"""
import argparse
import shutil
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--quantile',type=float,default=.35);ap.add_argument('--count',type=int,default=16);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006107);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);assert not state.get('args',{}).get('qat',False)
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=False
    d=m.load_data();rows=d['training'][::35][:6];calibration=[]
    for stage in range(1,4):
        node=f'res{stage}_conv1.pw';samples=[]
        for row in rows:
            z=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)[node]
            samples.append(z[0,:,::3,::3].flatten(1))
        z=torch.cat(samples,1);linear=((z>0)&(z<127)).float().mean(1);std=z.std(1)
        key=f'res{stage}_conv2__dw';nextw=net.weight[key]
        influence=net.weight[f'res{stage}_conv2__pw'].flatten(2)[:,:,0].norm(dim=0)*nextw.abs().sum((1,2,3))*std
        eligible=[j for j in range(32) if linear[j]>.99 and std[j]>.5]
        selected=sorted(eligible,key=lambda j:float(influence[j]))[:a.count]
        shifts=torch.quantile(z,a.quantile,dim=1)
        record=[]
        for c in selected:
            q=shifts[c];net.bias[node.replace('.','__')][c]-=q
            net.bias[key][c]+=q*nextw[c].sum()
            record.append(dict(channel=c,shift=float(q),before_linear=float(linear[c]),std=float(std[c]),influence=float(influence[c])))
        after=[]
        for row in rows:
            z=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)[node]
            after.append(((z[:,selected]>0)&(z[:,selected]<127)).float().mean((0,2,3)))
        rates=torch.stack(after).mean(0)
        for r,v in zip(record,rates):r['after_linear']=float(v)
        calibration.append(dict(stage=stage,records=record))
        print('RECENTER_STAGE',stage,len(selected),float(rates.mean()) if len(selected) else None,flush=True)
    # A direct numerical check of the DC compensation on the unchanged regime.
    w=torch.randn(3,1,3,3,device='cuda')*.1;shift=torch.tensor([1.,2.,3.],device='cuda')
    x=torch.rand(1,3,32,32,device='cuda')+shift[None,:,None,None]+1
    old=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),w,groups=3)
    new=F.conv2d(F.pad(F.relu(x-shift[None,:,None,None]),(1,1,1,1),mode='replicate'),w,shift*w.sum((1,2,3)),groups=3)
    compensation_error=float((old-new).abs().max());assert compensation_error<1e-5
    for key in net.weight:
        if key.startswith('up') or key=='rgb_head':
            assert torch.equal(net.weight[key].cpu(),state['model']['weight.'+key].cpu())
            assert torch.equal(net.bias[key].cpu(),state['model']['bias.'+key].cpu())
    torch.save(dict(model=net.state_dict(),qat=False,step=0,contract_sha256=m.t.r.CONTRACT_SHA,parent=a.warm,
        nonlinear_recenter=True),dest/'initial.pt');m.t.preview(net,d['validation'],dest/'preview')
    shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(args=vars(a),parent_sha256=file_hash(a.warm),calibration=calibration,
        training_ids=[r['id'] for r in rows],compensation_unit_error=compensation_error,frozen_decoder_unchanged=True,
        added_operators=0,style_success=False,code_sha256=file_hash(__file__)))
    print('RECENTER_COMPLETE',dest,compensation_error,flush=True)


if __name__=='__main__':main()
