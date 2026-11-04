"""A coefficient-only test of adjacent Y predicate consensus.

Mix adjacent cumulative predicates before their existing hard decisions. This
uses up1_low_pw entries, not a spatial filter or a new operation. The convex
row-stochastic matrix preserves every constant thermometer code after g11;
the full real-byte replay, real images and ideal contour width test whether
it also helps actual source predictions. A failed visual result is retained.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_monotone_pair_decoder import make_board
from .build_b1_thirty_binary_regions import code_bits
from .audit_b1_rim_tolerance import rim_terms
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,side):
    assert side in (0,.125,.25,.375)
    key='up1_low_pw';net.weight[key][:14].zero_();net.bias[key][:14].zero_()
    for i in range(14):
        for j,value in ((i,1-2*side),(max(0,i-1),side),(min(13,i+1),side)):
            net.weight[key][i,j,0,0]+=value
    net.up1_low_pw_m[:,:14].fill_(8192)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);assert state.get('direct_luminance_planes')
    out=Path(a.out);out.mkdir(exist_ok=False);rows=m.load_data(a.data)['validation']
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2));palette=fragment(base,z,'up1_low_dw')['rgb_head']
    boards=[{r['id']:Image.fromarray(m.t.r.read(r['target'])) for r in rows}];records=[]
    for side in (0,.125,.25,.375):
        net=copy.deepcopy(base);install(net,side)
        assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'],palette)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');images={}
        for row in rows:
            tensor=lambda path:torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
            x=tensor(row['path']);t=tensor(row['target']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            nodes=net(x,all_nodes=True);other,_=actual(net.round(x*128),True)
            assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            p=nodes['rgb_head']/128;rim,_=rim_terms(p,t,labels,2);_,v=reference_boundary_profile(p,t,labels)
            d=(p-t)*128
            records.append(dict(side=side,id=row['id'],mae=float(d.abs().mean()),mean_luminance_error=float((d[:,0]+2*d[:,1]+d[:,2]).mean()/4),
                tight_rim=float(rim),profile_shape=float(v['rgb_profile_shape']),profile_contrast=float(v['rgb_profile_contrast'])))
            images[row['id']]=m.t.r.c.rgb_image(p[0]);images[row['id']].save(out/f'side{side}_{row["id"]}.png')
        boards.append(images)
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
        saved.update(model=copy.deepcopy(net.state_dict()),luma_adjacent_consensus=side,constant_palette_preserved=1215,visual_acceptance=False)
        torch.save(saved,out/f'side{side}.pt')
        print('LUMA_CHANNEL_CONSENSUS',side,records[-len(rows):],flush=True)
    make_board(boards,['目标','原独立阈值','相邻系数1/8','相邻系数1/4','相邻系数3/8'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,all_nodes_integer_error=0,constant_palette_error=0,bytes=20480,
        extra_inference_operations=0,visual_acceptance=False,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
