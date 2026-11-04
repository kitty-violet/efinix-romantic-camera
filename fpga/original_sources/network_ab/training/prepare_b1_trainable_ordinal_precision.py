"""Make small learned changes representable in newly unfrozen fixed layers.

The old hand-built vote used integer-size weight steps. Refine only existing
Q17 row multipliers, preserving the current integer function before training.
Exact DW factorization and bounded pointwise rounding proofs are checked.
"""
import argparse,copy
from pathlib import Path
import torch
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def refine(net):
    key='res3_conv2__dw';assert (getattr(net,key+'_m')==8738).all()
    getattr(net,key+'_m').fill_(257)  #8738=34*257, old max3 becomes102.
    # Effective coefficients remain exactly the same; export changes units.
    key='res3_conv2__pw';scale=(getattr(net,key+'_m')/131072).float().flatten()
    oldw=net.round(net.weight[key]/scale[:,None,None,None]);oldb=net.round(net.bias[key]/scale)
    assert (oldw.abs()<=3).all() and torch.equal(oldw,torch.diag(torch.cat((torch.full((30,),3.),torch.zeros(2))))[:,:,None,None])
    net.weight[key].copy_(oldw);net.bias[key].copy_(oldb);getattr(net,key+'_m').fill_(4096)
    key='up1_low_dw';scale=(getattr(net,key+'_m')/131072).float().flatten()
    w=net.round(net.weight[key]/scale[:,None,None,None]);b=net.round(net.bias[key]/scale)
    assert w.abs().max()<=1 and b.abs().max()==0
    net.weight[key].copy_(w);net.bias[key].copy_(b);getattr(net,key+'_m').fill_(2048)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);net=copy.deepcopy(base);dest=Path(a.out);dest.mkdir(exist_ok=False);refine(net)
    # The two changed effective point kernels have identical clipped integer
    # output over the entire legal signed input set, not only validation crops.
    q=torch.arange(-128,128).float()
    for gain,bias in ((3,-127),(1,0)):
        old=net.round((q*gain+bias).double()*131071/131072).clamp(-128,127)
        new=(q*gain+bias).clamp(-128,127);assert torch.equal(old,new)
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    for row in m.load_data(a.data)['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
        old=base(x/128,all_nodes=True);now=net(x/128,all_nodes=True);replay,_=actual(x,True)
        assert all(torch.equal(v,old[k]) and torch.equal(v,replay[k]) for k,v in now.items())
    saved=copy.deepcopy(state);saved.update(model=net.state_dict(),trainable_ordinal_precision=True,visual_acceptance=False)
    torch.save(saved,dest/'candidate.pt');write_json(dest/'review.json',dict(source_all_six720_all_nodes_error=0,
      exact_dw_factor=34,pointwise_all256_inputs_error=0,bytes=20480,extra_operators=0,
      scales={'res3_conv2.dw':257/131072,'res3_conv2.pw':1/32,'up1_low_dw':1/64},
      source_sha256=file_hash(a.checkpoint),visual_acceptance=False))
    print('TRAINABLE_ORDINAL_PRECISION_PASS DW_exact/256point_inputs/6x720unchanged/replay/20KiB',flush=True)

if __name__=='__main__':main()
