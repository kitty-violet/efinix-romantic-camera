"""Separate Y geometry, chroma and co-located predicate disagreement.

All replacements explicitly use target data and are diagnosis ONLY. They are
not student predictions, deployable weights, or images suitable for delivery.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_monotone_pair_decoder import make_board
from .train_b1_luma_chroma_interfaces import luminance_codes,chroma
from .b1_joint_interface_targets import joint_interface_fields
from .audit_b1_rim_tolerance import rim_terms
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def same_active_mean(p,t):
    ids=F.pad((torch.diff(t,dim=1).abs()>.25).long().cumsum(1),(0,0,0,0,1,0))
    count=torch.zeros_like(t).scatter_add(1,ids,torch.ones_like(t)).clamp_min(1)
    mean=(torch.zeros_like(p).scatter_add(1,ids,p)/count).gather(1,ids)
    mask=(count.gather(1,ids)>=2)&(t>4)&(t<123)
    return torch.where(mask,mean,p)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    modes=('actual','known_Y','known_C','shared_Y_only','known_YC');boards=[{} for _ in range(6)];records=[]
    for row in m.load_data(a.data)['validation']:
        tensor=lambda path:torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
        x=tensor(row['path']);t=tensor(row['target']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        nodes=net(x,all_nodes=True);truth=joint_interface_fields(luminance_codes(t,labels,net.round))[:,:14]
        c=chroma(t)*128;c[:,0]=(c[:,0]+36)*8/9;c[:,1]=(c[:,1]+24)*8/6
        c=F.avg_pool2d(c,4).clamp(0,127);z=nodes['up1_low_pw'];boards[0][row['id']]=Image.fromarray(m.t.r.read(row['target']))
        for j,mode in enumerate(modes,1):
            u=z.clone()
            if mode in ('known_Y','known_YC'):u[:,:14]=truth
            if mode in ('known_C','known_YC'):u[:,14:16]=c
            if mode=='shared_Y_only':u[:,:14]=same_active_mean(z[:,:14],truth)
            rgb=fragment(net,net.round(u),'up1_low_pw')['rgb_head']
            if mode=='actual':assert torch.equal(rgb,nodes['rgb_head'])
            p=rgb/128;rim,_=rim_terms(p,t,labels,2);_,terms=reference_boundary_profile(p,t,labels)
            records.append(dict(id=row['id'],mode=mode,mae=float((p-t).abs().mean()*128),tight_rim=float(rim),
                profile_shape=float(terms['rgb_profile_shape']),profile_contrast=float(terms['rgb_profile_contrast'])))
            im=m.t.r.c.rgb_image(p[0]);boards[j][row['id']]=im;im.save(out/f'{mode}_{row["id"]}.png')
        print('LUMA_READOUT_ORIGIN',row['id'],records[-5:],flush=True)
    make_board(boards,['目标','实际预测','仅替换正确Y：诊断','仅替换正确C：诊断','Y共边对齐：诊断','已知Y+C：诊断'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,ground_truth_interventions=True,not_deployable=True,
        visual_acceptance=False,actual_fragment_equal=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__)))


if __name__=='__main__':main()
