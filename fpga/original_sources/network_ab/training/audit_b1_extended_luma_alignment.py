"""Diagnostic target-labelled alignment over the measured displacement range.

Unlike the old8px active mask, include up to32px of the same interface.
These images use known target labels and MUST NOT be delivered as predictions.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .audit_b1_luma_readout_origin import same_active_mean
from .audit_b1_rim_tolerance import rim_terms
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,net=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False);records=[];boards=[{} for _ in range(5)]
    for row in m.load_data(a.data)['validation']:
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target'])
        x=torch.from_numpy(source.transpose(2,0,1).copy())[None].float()/128-1;t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        with np.load(row['native_luma_codes']) as z:code=torch.from_numpy(z['codes'].astype(np.int64))[None]
        nodes=net(x,all_nodes=True);boards[0][row['id']]=Image.fromarray(target)
        for j,(name,gain,node) in enumerate((('actual',0,'up1_low_pw'),('narrow',8,'up1_low_pw'),('wide180',2,'up1_low_pw'),('wide360',2,'up1_high_dw')),1):
            u=nodes[node].clone()
            if gain:
                truth=joint_interface_fields(code,gain)[:,:14]
                if node=='up1_high_dw':truth=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_pw')[node][:,:14]
                u[:,:14]=same_active_mean(u[:,:14],truth)
            rgb=fragment(net,net.round(u),node)['rgb_head'];p=rgb/128
            if not gain:assert torch.equal(rgb,nodes['rgb_head'])
            rim,_=rim_terms(p,t,labels,2);records.append(dict(id=row['id'],mode=name,mae=float((p-t).abs().mean()*128),tight_rim=float(rim)))
            im=m.t.r.c.rgb_image(p[0]);im.save(out/f'{name}_{row["id"]}.png');boards[j][row['id']]=im
        print('EXTENDED_LUMA_ALIGNMENT',row['id'],records[-4:],flush=True)
    make_board(boards,['目标','实际预测','已知窄带：诊断','已知宽带180：诊断','已知宽带360：诊断'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,ground_truth_interventions=True,not_deployable=True,visual_acceptance=False,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__)))


if __name__=='__main__':main()
