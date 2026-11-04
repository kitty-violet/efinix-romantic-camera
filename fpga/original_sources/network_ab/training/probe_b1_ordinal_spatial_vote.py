"""Use two existing layers for a rank-preserving spatial colour decision.

Spatial voting occurs on30 nested ordinal predicates, before binary packing.
This differs from voting on independently predicted/remainder bits. For binary
predicates the two layers implement exact3x3 majority. Soft inputs are not claimed
to be an exact median, nor is a global minimum connected area guaranteed.
"""
import argparse,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_exact_complement_coverage import install as exact_install
from .build_b1_thirty_binary_regions import code_bits,GROUPS30
from .audit_b1_common_coordinate_boundaries import hinges
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install_vote(net):
    key='res3_conv2__dw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(14563)
    net.weight[key][:30,0].fill_(14563/131072)
    key='res3_conv2__pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    for i in range(30):net.weight[key][i,i,0,0]=9*131071/131072
    net.bias[key][:30]=-508*131071/131072


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False);setup=copy.deepcopy(state['binary_coverage720'])
    exact_install(net,7);install_vote(net)
    bits=((torch.arange(512)[:,None]>>torch.arange(9)[None])&1).float();z=torch.zeros(512,32,3,3);z[:,0]=bits.reshape(-1,3,3)*127
    # res3 PW is signed; the already-existing up1 DW ReLU completes clipping.
    out=fragment(net,z,'res3_conv1.pw');assert torch.equal(out['up1_low_dw'][:,0,1,1],(bits.sum(1)>=5).float()*127)
    code=torch.arange(1215)[:,None,None];z=F.pad(code_bits(code)*127,(0,0,0,0,0,2));out=fragment(net,z,'res3_conv1.pw')['rgb_head'][:,:,0,0]
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127)
    assert torch.equal(out,palette)
    coordinates=torch.rand(2,3,48,64)*torch.tensor([14,8,8])[None,:,None,None];z=net.round(hinges(coordinates));out=fragment(net,z,'res3_conv1.pw')['up1_low_dw']
    assert all((torch.diff(out[:,lo:hi],dim=1)<=0).all() for lo,hi in GROUPS30)
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    for row in m.load_data(a.data)['validation']:
        src=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
        nodes=net(src/128,all_nodes=True);other,_=actual(src,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        m.t.r.c.rgb_image(nodes['rgb_head'][0]/128).save(dest/(row['id']+'_output.png'))
    setup.update(gain=7,score_center=63.5,score_endpoints=[0,127],exact_integer_complement=True,ordinal_spatial_vote=True)
    saved=copy.deepcopy(state);saved.update(model=net.state_dict(),binary_coverage720=setup,ordinal_spatial_vote=True,step=0,visual_acceptance=False)
    torch.save(saved,dest/'candidate.pt');write_json(dest/'review.json',dict(exhaustive512_binary_neighbourhoods=True,ordinal_rank_preserved=True,
        constant1215_palette_error=0,actual_all_six720_node_error=0,byte_size=20480,extra_operators=0,minimum_area_guarantee=False,
        visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))
    print('ORDINAL_SPATIAL_VOTE_PASS 512patterns/rank/1215colours/6x720_integer/20KiB',flush=True)


if __name__=='__main__':main()
