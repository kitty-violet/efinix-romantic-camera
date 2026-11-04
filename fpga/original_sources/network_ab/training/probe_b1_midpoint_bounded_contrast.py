"""Correct the omitted midpoint term, with exact integer local bounds.

v=L+clip(g*(u-(H+L)/2)+(H-L)/2,0,H-L).
Two existing ReLU fields use floor(A),ceil(B), so g*relu(floor(A)-ceil(B))
cannot exceed H-L. No hidden float postfilter is used. Quantized conservatism
is measured explicitly; these are designed coefficients, NOT trained output.
"""
import argparse,copy
from pathlib import Path
import torch
from .probe_b1_bounded_source_shock import install as old_install
from .probe_b1_encoder_gauge import load
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,gain):
    old_install(net,gain,True)
    def point(k,mapping,bias=None,quantum=1/64):
        net.weight[k].zero_();net.bias[k].zero_();getattr(net,k+'_m').fill_(round(quantum*131072))
        for o,terms in mapping.items():
            for i,v in terms.items():net.weight[k][o,i,0,0]=v
        if bias:
            for o,v in bias.items():net.bias[k][o]=v
    # Input H,L,u,Co,Cg. Floor/ceil use1/64 exact representable tie offsets.
    ah=(gain-1)/(2*gain);al=(gain+1)/(2*gain)
    point('res3_conv1__pw',{0:{2:1,0:-ah,1:-al},1:{2:1,0:-al,1:-ah},2:{1:1},3:{3:1},4:{4:1}}, {0:-31/64,1:31/64})
    # Signed correction D; the following existing DW ReLU clamps it at0.
    point('res3_conv2__pw',{0:{0:gain,1:-gain},1:{2:1},2:{3:1},3:{4:1}},quantum=1/16)
    k='up1_low_dw';net.weight[k].zero_();net.bias[k].zero_();net.up1_low_dw_m.fill_(2048);net.weight[k][:4,0,1,1]=1
    point('up1_low_pw',{i:{0:8,1:8} for i in range(14)}|{14:{2:1},15:{3:1}},
          {i:63.5-8*(12+8*i) for i in range(14)},quantum=1/8)


def regression():
    # Exhaust all integer half-range triples, including u outside[L,H].
    from .qat_b1_deploy_preview import Round
    rnd=Round();u=torch.arange(128)[:,None,None].float();lo=torch.arange(128)[None,:,None].float();hi=torch.arange(128)[None,None,:].float();valid=lo<=hi
    for g in (2,4):
        ah=(g-1)/(2*g);al=(g+1)/(2*g)
        aa=rnd(u-ah*hi-al*lo-31/64).clamp(0,127);bb=rnd(u-al*hi-ah*lo+31/64).clamp(0,127)
        d=(g*(aa-bb)).clamp(0,127);v=lo+d
        assert not (((v<lo)|(v>hi))&valid).any()
        ideal=lo+(g*(u-(hi+lo)/2)+(hi-lo)/2).clamp_min(0).minimum(hi-lo)
        error=(v-ideal).abs()[valid.expand_as(v)].max();assert error<=2*g
        print('MIDPOINT_RANGE_EXHAUSTIVE',g,'all_integer_triples_in_range',float(error),flush=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);regression();state,base=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False);boards=[];records=[]
    for gain in (2,4):
        net=copy.deepcopy(base);install(net,gain);raw=net.export();assert len(raw)==20480
        runtime=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');shots={}
        for row in m.load_data(a.data)['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=runtime(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            hi,lo=nodes['res2_add'][:,0],nodes['res2_add'][:,1];u=nodes['up1_low_dw'][:,0]+nodes['up1_low_dw'][:,1]
            assert (u>=lo).all() and (u<=hi).all() and (torch.diff(nodes['up1_high_dw'][:,:14],dim=1)<=0).all()
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(out/f'g{gain}_{row["id"]}.png');shots[row['id']]=im
            records.append(dict(gain=gain,id=row['id'],actual_all_nodes_error=0,local_half_luma_range_error=0))
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            designed_midpoint_bounded_contrast=True,learned=False,gain=gain,visual_acceptance=False)
        for k in ('optimizer','rng','torch_rng','cuda_rng'):saved.pop(k,None)
        torch.save(saved,out/f'g{gain}.pt');boards.append(shots);print('MIDPOINT_BOUNDED',gain,'6nodes0/exact_range0/order/20KiB',flush=True)
    make_board(boards,['中点修正·有界对比2','中点修正·有界对比4'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),designed_not_learned=True,new_operators=0,visual_acceptance=False))


if __name__=='__main__':main()
