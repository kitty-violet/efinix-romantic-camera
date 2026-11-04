"""Actual-byte quality checks for14Y or radixY plus continuous chroma.

Co/Cg are no longer independent ordinal classifiers. Do not report the old
30-field distance metric as this model's training target. Colour fringes use
the original labelled region means, not the obsolete nine-level colour table.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .train_b1_luma_chroma_interfaces import luminance_codes
from .b1_joint_interface_targets import joint_interface_fields
from .train_b1_ordered_distance_rims import distance_objective
from .audit_b1_rim_tolerance import rim_terms
from .audit_b1_colour_fringe_origin import fringe,two_colour_support
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_target_chain import region_means,adjacency,lab
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash,contract_guard
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--reserved',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);contract_guard();state,net=load(a.checkpoint);assert state.get('radix_luma_continuous_chroma')
    out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data);rows=list(data['validation'])
    if a.reserved:rows+=data['reserved_test']
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    supervision_node=state.get('luma_supervision_node','up1_low_dw')
    if supervision_node=='up1_low_pw':
        from .train_b1_trainable_luma_readout import backend_palette
        palette=backend_palette(net)
    else:
        z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2));palette=fragment(net,z,'up1_low_dw')['rgb_head'][:,:,0,0]
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    expected=(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127);assert torch.equal(palette,expected)
    records=[];boards=[{},{}];tensor=lambda q:torch.from_numpy(q.transpose(2,0,1).copy())[None].float()/128-1
    for row in rows:
        for key,sha in (('path','sha256'),('target','target_sha256'),('region','region_sha256')):assert file_hash(row[key])==row[sha]
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target']);labels=m.load_region(row['region']).astype(np.int64)
        x=tensor(source);t=tensor(target);ll=torch.from_numpy(labels)[None];nodes=net(x,all_nodes=True);other,_=actual(net.round(x*128),True)
        assert all(torch.equal(v,other[k]) for k,v in nodes.items());p=nodes['rgb_head']/128;rgb=nodes['rgb_head'][0].permute(1,2,0).numpy()+128
        means,count=region_means(target,labels);ca,cb,valid=two_colour_support(labels,means)
        rim,_=rim_terms(p,t,ll,2);_,profile=reference_boundary_profile(p,t,ll)
        if state.get('native_region_luma_labels'):
            assert file_hash(row['native_luma_codes'])==row['native_luma_codes_sha256']
            with np.load(row['native_luma_codes']) as z:code=torch.from_numpy(z['codes'].astype(np.int64))[None]
        else:code=luminance_codes(t,ll,net.round)
        full,band=distance_objective(nodes[supervision_node][:,:14],joint_interface_fields(code)[:,:14])
        predicted,_=region_means(rgb,labels);aa,bb,length=adjacency(labels);wanted=lab(means)[aa]-lab(means)[bb];got=lab(predicted)[aa]-lab(predicted)[bb]
        strong=np.linalg.norm(wanted,axis=1)>=5;den=max(1,length[strong].sum())
        d=rgb-target.astype(np.float32);core=(F.max_pool2d(torch.from_numpy(np.pad((labels[:,1:]!=labels[:,:-1]),((0,0),(1,0)))|np.pad(labels[1:]!=labels[:-1],((1,0),(0,0))))[None,None].float(),5,1,2)==0)[0,0].numpy()
        roi=[]
        for rid,(x0,y0,x1,y1),title in m.t.r.c.CROPS:
            if rid!=row['id']:continue
            dd=d[y0:y1,x0:x1][core[y0:y1,x0:x1]]
            roi.append(dict(region=title,core_pixels=len(dd),bias_Co=float(((dd[:,0]-dd[:,2])/2).mean()),bias_Cg=float(((2*dd[:,1]-dd[:,0]-dd[:,2])/4).mean())))
        rec=dict(id=row['id'],split=row['split'],mae=float(np.abs(d).mean()),luminance_bias=float((.25*d[:,:,0]+.5*d[:,:,1]+.25*d[:,:,2]).mean()),
            tight_rim=float(rim),profile_shape=float(profile['rgb_profile_shape']),profile_contrast=float(profile['rgb_profile_contrast']),
            luma_interface_full=float(full),luma_interface_band=float(band),original_region_colour_fringe=fringe(rgb,ca,cb,valid),roi_chroma_bias=roi,
            collapsed_fraction=float(length[(np.linalg.norm(got,axis=1)<2)&strong].sum()/den),reversed_fraction=float(length[((wanted*got).sum(1)<0)&strong].sum()/den))
        records.append(rec);im=Image.fromarray(rgb.astype(np.uint8));im.save(out/(row['id']+'_integer720.png'))
        boards[0][row['id']]=Image.fromarray(target);boards[1][row['id']]=im
        write_json(out/'review.json',dict(records=records,all_checked_nodes_integer_error=0,constant1215_error=0,bytes=20480,checkpoint_sha256=file_hash(a.checkpoint),
            source_sha256=file_hash(__file__),fringe_reference='original region means, two-region5x5 support',luma_supervision='14 shared-interface predicates',supervision_node=supervision_node,
            native_region_luma_labels=bool(state.get('native_region_luma_labels')),data_sha256=file_hash(a.data),
            reserved_previously_used_for_reconstruction=a.reserved,temporal_acceptance=False,visual_acceptance=False,complete=False))
        print('LUMA_CHROMA_QUALITY',row['id'],rec['mae'],rec['tight_rim'],flush=True)
    make_board(boards,['目标','实际20KiB重载输出'],out/'comparison.png')
    result=__import__('json').loads((out/'review.json').read_text(encoding='utf-8'));result['complete']=True;write_json(out/'review.json',result)
    print('LUMA_CHROMA_QUALITY_COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
