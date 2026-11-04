"""Bounded local contrast selection inside the fixed B1 graph.

Inspired by shock filtering's local range principle, not a reproduction of
the Osher/Rudin PDE. Existing DW/PW/ReLU stages find four-neighbour bounds,
then implement L+clip(k*(u-(H+L)/2),0,H-L) with two positive parts.
The brightness changes are bounded; no negative unsharp convolution follows
the output. These are explicit designed coefficients, not learned weights.
"""
import argparse,copy
from pathlib import Path
import torch
from .probe_b1_encoder_gauge import load
from .probe_b1_shared_positive_regions import install as initial_install
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,gain,directional_mean):
    initial_install(net,0)
    def point(k,mapping,bias=None,quantum=1/32):
        net.weight[k].zero_();net.bias[k].zero_();getattr(net,k+'_m').fill_(round(quantum*131072))
        for out,terms in mapping.items():
            for inp,value in terms.items():net.weight[k][out,inp,0,0]=value
        if bias:
            for out,value in bias.items():net.bias[k][out]=value
    def identity(k,n):
        net.weight[k].zero_();net.bias[k].zero_();getattr(net,k+'_m').fill_(2048)
        net.weight[k][:n,0,1,1]=1
    point('down1__pw',{0:{0:.25,1:.5,2:.25},1:{0:1.,2:-1.},2:{0:-.75,1:1.5,2:-.75}},{1:64,2:64})
    point('down2__pw',{i:{0:1} for i in range(5)}|{5:{1:1},6:{2:1}})
    identity('res1_conv1__dw',7);w=net.weight['res1_conv1__dw'];w[:4].zero_()
    if directional_mean:
        w[0,0,:,2]=torch.tensor([1,2,1])/4;w[1,0,:,0]=torch.tensor([1,2,1])/4
        w[2,0,0,:]=torch.tensor([1,2,1])/4;w[3,0,2,:]=torch.tensor([1,2,1])/4
    else:
        for i,(y,x) in enumerate(((1,2),(1,0),(0,1),(2,1))):w[i,0,y,x]=1
    point('res1_conv1__pw',{0:{0:1,1:-1},1:{2:1,3:-1},2:{0:1},3:{1:1},4:{2:1},5:{3:1},6:{4:1},7:{5:1},8:{6:1}})
    identity('res1_conv2__dw',9)
    point('res1_conv2__pw',{0:{3:1,0:1},1:{2:1,0:-1},2:{5:1,1:1},3:{4:1,1:-1},4:{6:1},5:{7:1},6:{8:1}})
    identity('res2_conv1__dw',7)
    point('res2_conv1__pw',{0:{0:1,2:-1},1:{1:1,3:-1},2:{2:1},3:{1:1},4:{4:1},5:{5:1},6:{6:1}})
    identity('res2_conv2__dw',7)
    point('res2_conv2__pw',{0:{2:1,0:1},1:{3:1,1:-1},2:{4:1},3:{5:1},4:{6:1}})
    identity('res3_conv1__dw',5)
    point('res3_conv1__pw',{0:{2:1,0:-.5,1:-.5},1:{2:1,0:-.5-1/gain,1:-.5+1/gain},2:{1:1},3:{3:1},4:{4:1}})
    identity('res3_conv2__dw',5)
    point('res3_conv2__pw',{i:{2:8,0:8*gain,1:-8*gain} for i in range(14)}|{14:{3:1},15:{4:1}},
          {i:63.5-8*(12+8*i) for i in range(14)},quantum=.5)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    boards=[];records=[];rows=m.load_data(a.data)['validation']
    for gain,mean in ((2,False),(4,False),(4,True)):
        name=f'g{gain}_mean{int(mean)}';net=copy.deepcopy(base);install(net,gain,mean);raw=net.export();assert len(raw)==20480
        runtime=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=runtime(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            hi,lo=nodes['res2_add'][:,0],nodes['res2_add'][:,1];v=nodes['res3_conv1.pw'];u=v[:,2]+gain*(v[:,0]-v[:,1])
            bound_error=float(torch.maximum(u-hi,lo-u).clamp_min(0).max());assert bound_error<=gain/2
            assert (torch.diff(nodes['up1_high_dw'][:,:14],dim=1)<=0).all()
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(out/f'{name}_{row["id"]}.png');shots[row['id']]=im
            records.append(dict(mode=name,id=row['id'],local_half_luma_range_error=bound_error,actual_all_nodes_error=0))
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            designed_bounded_source_shock=True,gain=gain,directional_mean=mean,learned=False,visual_acceptance=False)
        for k in ('optimizer','rng','torch_rng','cuda_rng'):saved.pop(k,None)
        torch.save(saved,out/f'{name}.pt');boards.append(shots);print('BOUNDED_SOURCE_SHOCK',name,'6_nodes0/range/order/20KiB',flush=True)
    make_board(boards,['局部有界对比2','局部有界对比4','方向平均+有界对比4'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),dependency_sha256=file_hash(Path(__file__).with_name('probe_b1_shared_positive_regions.py')),
        reference='https://epubs.siam.org/doi/10.1137/0727053',not_exact_pde=True,new_operators=0,designed_not_learned=True,visual_acceptance=False))


if __name__=='__main__':main()
