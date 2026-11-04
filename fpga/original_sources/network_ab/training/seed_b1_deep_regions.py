"""Predict region coverage with the learned encoder before fixed region cleanup.

Unlike the earlier seed, predictions use the stem through res2.conv2.dw, not
only four shallow colour convolutions. Residual metadata is explicit.
"""
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_region_majority import build as decoder
from .probe_b1_dense_levels import oracle_records
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible

THRESHOLDS=[(0,v) for v in (8,18,30,44,60,78,98,118)]+[(1,v) for v in (49,59,69,79)]+[(2,v) for v in (49,59,69,79)]
INCREMENTS=(9,11,13,15,17,19,20,15)


@torch.no_grad()
def main():
    configure_workspace();set_reproducible(2026100687);dest=m.OUT/'deep-regions-initial';dest.mkdir(exist_ok=False)
    parent=m.t.OUT/'clear-regions/best.pt';net=m.t.r.initialize()
    net.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['model']);net.quantized=True
    d=m.load_data();rows=d['training'][::max(1,len(d['training'])//24)][:24]
    xx=torch.zeros(33,33,device='cuda',dtype=torch.float64);xy=torch.zeros(33,3,device='cuda',dtype=torch.float64)
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        feature=net(x,all_nodes=True)['res2_conv2.dw']
        z=F.avg_pool2d((target+1)*64,4);r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
        colour=torch.cat(((r+2*g+b)/4,(r-b)/2+64,(-r+2*g-b)/4+64),1)
        a=feature[0,:,::2,::2].flatten(1).T.double()/64;a=torch.cat((a,torch.ones_like(a[:,:1])),1)
        y=colour[0,:,::2,::2].flatten(1).T.double();xx+=a.T@a;xy+=a.T@y
    reg=torch.eye(33,device='cuda',dtype=torch.float64)*xx.diag().mean()*.00001;reg[-1,-1]=0
    fit=torch.linalg.solve(xx+reg,xy).float()
    # Copy only the late renderer, then a gentle final180p coverage cleanup.
    tail=decoder(passes=0,geometry=4.,outline=.5)
    reached=False
    for key in net.weight:
        if key=='res3_conv1__dw':reached=True
        if reached:
            net.weight[key].copy_(tail.weight[key]);net.bias[key].copy_(tail.bias[key])
            getattr(net,key+'_m').copy_(getattr(tail,key+'_m'))
    net.package['residual_encoding']['res2_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    net.package['residual_encoding']['res3_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    w,b=net.weight['res2_conv2__pw'],net.bias['res2_conv2__pw'];w.zero_();b.zero_()
    gain=8.
    for j,(c,threshold) in enumerate(THRESHOLDS):
        w[j,:,0,0]=fit[:-1,c]/64*gain;b[j]=64+(fit[-1,c]-threshold)*gain
    # Two spatial averaging steps on level-set coverage; low-gradient interior
    # values are pulled towards0/127 by the following affine/clipping stage.
    kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for half in (1,2):
        key=f'res3_conv{half}__dw';net.weight[key].zero_();net.bias[key].zero_()
        for c in range(16):net.weight[key][c,0]=kernel
        key=f'res3_conv{half}__pw';net.weight[key].zero_();net.bias[key].zero_()
        for c in range(16):net.weight[key][c,c,0,0]=2;net.bias[key][c]=-63.5
    # Decoder uses the nonuniform brightness increments, while chroma stays
    # neutral-centred. Colour is reconstructed after region geometry.
    net.weight['up2_low_pw'][:3,:8,0,0]=w.new_tensor(INCREMENTS)[None]/127
    net.bias['up2_low_pw'][:3]-=3.28
    for c in range(3,7):net.weight['up2_low_pw'][c,:8,0,0]=w.new_tensor(INCREMENTS)/127;net.bias['up2_low_pw'][c]=4
    for key,w in net.weight.items():
        if key=='res2_conv2__pw' or key.startswith('res3') or key.startswith('up') or key=='rgb_head':
            peak=w.abs().flatten(1).amax(1).clamp_min(1e-5);scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
            getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    raw=net.export();(dest/'initial.bin').write_bytes(raw);actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw));errors=[]
    for row in d['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];a=net(x,all_nodes=True);b,_=actual(x*128,True)
        errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
    assert max(errors)==0
    params,residual=m.t.r.decode(raw);op,rp=oracle_records(params,residual)
    sample=np.random.default_rng(87).integers(-128,128,(12,12,3),dtype=np.int16)
    ref=m.t.r.execute_integer(m.t.r.contract_guard()[0],sample.tolist(),op,rp)
    got,_=actual(torch.tensor(sample.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32),True)
    error=max(float((v[0].permute(1,2,0).cpu()-torch.tensor(ref['tensors'][k])).abs().max()) for k,v in got.items())
    assert error==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=residual,contract_sha256=m.t.r.CONTRACT_SHA,
        linear_fit_training_ids=[r['id'] for r in rows],parent=str(parent)),dest/'initial.pt')
    m.t.preview(net,d['validation'],dest/'preview');shutil.copy2(__file__,dest/'seed_source.py')
    write_json(dest/'review.json',dict(all_node_error=max(errors),oracle_error=error,added_operators=0,sgd_steps=0,
        fixed_area_guarantee=False,parent_sha256=file_hash(parent),code_sha256=file_hash(__file__)))
    print('DEEP_REGION_INIT',max(errors),error,flush=True)


if __name__=='__main__':main()
