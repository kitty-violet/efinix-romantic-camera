"""Test low-side neighbourhood selection without amplified subtraction.

This contrasts a local range operation with the failed rounded strong-shock
operation. All variants are explicit B1 coefficients and remain diagnostics.
"""
import argparse,copy
from pathlib import Path
import torch
from .probe_b1_encoder_gauge import load
from .probe_b1_bounded_source_shock import install as shock_install
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,amount):
    shock_install(net,2,True)
    # Preserve the centre for a convex combination with the neighbour minimum.
    net.weight['res3_conv1__pw'][5,2,0,0]=1
    net.weight['res3_conv2__dw'][5,0,1,1]=1
    key='res3_conv2__pw';net.weight[key][:14].zero_()
    for i in range(14):
        net.weight[key][i,2,0,0]=8*amount
        net.weight[key][i,5,0,0]=8*(1-amount)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    boards=[];records=[]
    for amount in (0.,.5,1.):
        net=copy.deepcopy(base);install(net,amount);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={}
        for row in m.load_data(a.data)['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);reloaded,_=actual(net.round(x*128),True);assert all(torch.equal(v,reloaded[k]) for k,v in nodes.items())
            assert len(raw)==20480 and (torch.diff(nodes['up1_high_dw'][:,:14],dim=1)<=0).all()
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(out/f'amount{amount}_{row["id"]}.png');shots[row['id']]=im
            records.append(dict(amount=amount,id=row['id'],actual_all_nodes_error=0))
        s=copy.deepcopy(state);s.update(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            designed_local_low_side_selection=True,low_side_amount=amount,learned=False,visual_acceptance=False)
        for k in ('optimizer','rng','torch_rng','cuda_rng'):s.pop(k,None)
        torch.save(s,out/f'amount{amount}.pt');boards.append(shots);print('BOUNDED_SOURCE_RANGE',amount,'6_nodes0/order/20KiB',flush=True)
    make_board(boards,['不做局部选择','局部低侧混合一半','局部低侧选择'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),designed_not_learned=True,new_operators=0,visual_acceptance=False))


if __name__=='__main__':main()
