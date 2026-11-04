"""Source-only monotone morphology encoded as ordinary B1 coefficients.

All14 luminance masks share positive spatial filters and monotone pointwise
steps. This is a designed diagnostic, not a learned model or accepted result.
There is no source/target mask, global segmentation or postfilter at inference.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,rounds,saturation=1.125):
    def reset(k,quantum=1/64):
        net.weight[k].zero_();net.bias[k].zero_();getattr(net,k+'_m').fill_(round(quantum*131072))
    def identity(k,n=16):
        reset(k)
        if net.weight[k].shape[-1]==3:
            net.weight[k][:n,0,1,1]=1
        else:
            for i in range(n):net.weight[k][i,i,0,0]=1
    blur=torch.tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for k in net.weight:reset(k)
    # Stem stores unsigned half-range RGB; no new skip or high-res path.
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    net.weight['down1__dw'][:3,0]=blur
    reset('down1__pw',1/16)
    for i in range(14):
        net.weight['down1__pw'][i,:3,0,0]=torch.tensor([2.,4.,2.])
        net.bias['down1__pw'][i]=63.5-8*(12+8*i)
    net.weight['down1__pw'][14,:3,0,0]=torch.tensor([1.,0.,-1.])*saturation*8/9
    net.weight['down1__pw'][15,:3,0,0]=torch.tensor([-.5,1.,-.5])*saturation*8/6
    net.bias['down1__pw'][14:16]=64
    net.weight['down2__dw'][:16,0]=blur;identity('down2__pw')
    phase=0
    for stage in range(1,4):
        for half in (1,2):
            dw=f'res{stage}_conv{half}__dw';pw=f'res{stage}_conv{half}__pw'
            identity(dw);identity(pw)
            if phase<rounds:
                net.weight[dw][:14,0]=blur
                # Same centred saturating map on every ordered mask.
                reset(pw,1/32)
                for i in range(14):net.weight[pw][i,i,0,0]=2;net.bias[pw][i]=-63.5
                for i in (14,15):net.weight[pw][i,i,0,0]=1
            phase+=1
        # The package already supports this coefficient-only skip suppression.
        net.package['residual_encoding'][f'res{stage}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    identity('up1_low_dw');identity('up1_low_pw');net.weight['up1_high_dw'][:16,0]=torch.tensor([[1,2,1],[2,3,2],[1,2,1]])/15
    # Exactly represent the shared /15 kernel in the existing multiplier.
    net.up1_high_dw_m.fill_(round(131072/15));net.weight['up1_high_dw'][:16,0]=torch.tensor([[1,2,1],[2,3,2],[1,2,1]])*(round(131072/15)/131072)
    reset('up2_low_dw',131071/131072)
    net.weight['up2_low_dw'][:14,0,1,1]=11*131071/131072;net.bias['up2_low_dw'][:14]=-635*131071/131072
    net.weight['up2_low_dw'][14:16,0,1,1]=131071/131072
    reset('up2_low_pw',8256/131072)
    net.weight['up2_low_pw'][0,:14,0,0]=8256/131072
    net.weight['up2_low_pw'][1,14,0,0]=1;net.weight['up2_low_pw'][2,15,0,0]=1
    # Identity coefficients require a coarser per-row quantum than the Y sum.
    net.up2_low_pw_m[:,1:3].fill_(2048)
    net.weight['up2_high_dw'][:3,0]=blur
    reset('rgb_head',1/32)
    net.weight['rgb_head'][:,:3,0,0]=torch.tensor([[16,9,-6],[16,0,6],[16,-9,-6]])/8
    net.bias['rgb_head'][:]=torch.tensor([-136.,-160.,8.])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    rows=m.load_data(a.data)['validation'];boards=[];records=[]
    for rounds in (0,2,4):
        net=copy.deepcopy(base);install(net,rounds);raw=net.export();assert len(raw)==20480
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);other,_=actual(net.round(x*128),True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            # Ordered luminance maps must stay ordered at every point.
            assert (torch.diff(nodes['up1_high_dw'][:,:14],dim=1)<=0).all()
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()-128
            rec=dict(rounds=rounds,id=row['id'],mae=float((nodes['rgb_head']-target).abs().mean()),all_nodes_error=0)
            records.append(rec);im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(out/f'rounds{rounds}_{row["id"]}.png');shots[row['id']]=im
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            designed_shared_positive_regions=True,morphology_rounds=rounds,learned=False,visual_acceptance=False)
        for k in ('optimizer','rng','torch_rng','cuda_rng'):saved.pop(k,None)
        torch.save(saved,out/f'rounds{rounds}.pt');boards.append(shots);print('SHARED_POSITIVE_REGIONS',rounds,'6_all_nodes0/order/20KiB',flush=True)
    make_board(boards,['直接亮度层','共享单调处理2次','共享单调处理4次'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),
        designed_not_learned=True,source_only_inference=True,new_operators=0,bytes=20480,visual_acceptance=False))


if __name__=='__main__':main()
