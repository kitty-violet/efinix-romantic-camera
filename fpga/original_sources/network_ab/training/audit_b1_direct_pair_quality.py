"""Actual-byte and visual diagnostics for ordered-field B1 checkpoints."""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_monotone_pair_decoder import GROUPS15,make_board
from .build_b1_thirty_binary_regions import GROUPS30,code_bits
from .train_b1_ternary_regions import region_codes
from .audit_b1_white_rims_grid import measures
from .audit_b1_target_chain import region_means,adjacency,lab
from .runtime import write_json,file_hash
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .train_b1_coincident_distance import coincident_loss
from . import train_b1_minimum_area as m

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state.get('ordinal_pair_setup') or state['binary_coverage720'];rows=m.load_data(a.data)['validation']
    matrix=torch.tensor(setup['matrix']).float();bias=torch.tensor(setup['bias']).float()
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();palette=(levels@matrix.T+bias).clamp(-128,127)
    pk=tuple(map(tuple,(palette+128).tolist()));z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'][:,:,0,0],palette)
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    boards=[{}, {}, {}, {}];records=[]
    for row in rows:
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target']);labels=m.load_region(row['region']).astype(np.int64)
        x=torch.from_numpy(source.transpose(2,0,1).copy())[None].float()-128;t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float()
        nodes=net(x/128,all_nodes=True);replay,_=actual(x,True);assert all(torch.equal(v,replay[k]) for k,v in nodes.items())
        output=nodes['rgb_head']+128;stats,mask=measures(output,t,x+128)
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        truth=F.avg_pool2d(code_bits(code),4);fields=nodes['up1_low_dw'][:,:30]/127
        mixed=(truth>0)&(truth<1);error=(fields-truth).abs()
        distance=distance_fields(code,float(state.get('ordered_distance_gain',8)))
        supervised=distance/127 if state.get('ordered_distance_gain') else truth
        _,profile=reference_boundary_profile(nodes['rgb_head']/128,t/128-1,torch.from_numpy(labels)[None])
        coords=torch.stack([fields[:,lo:hi].sum(1) for lo,hi in GROUPS30],1)
        view=torch.einsum('oc,bchw->bohw',matrix,coords)+bias[None,:,None,None]
        view=F.interpolate(view,size=(720,1280),mode='nearest').clamp(-128,127)
        means,count=region_means(target,labels);predicted,_=region_means(output[0].permute(1,2,0).numpy(),labels)
        aa,bb,length=adjacency(labels);wanted=lab(means)[aa]-lab(means)[bb];got=lab(predicted)[aa]-lab(predicted)[bb]
        strong=np.linalg.norm(wanted,axis=1)>=5;den=max(1,length[strong].sum())
        record=dict(id=row['id'],**stats,mae=float((output-t).abs().mean()),
          collapsed_fraction=float(length[(np.linalg.norm(got,axis=1)<2)&strong].sum()/den),
          reversed_fraction=float(length[((wanted*got).sum(1)<0)&strong].sum()/den),
          ordinal_area_mae=float(error.mean()),mixed_ordinal_area_mae=float(error[mixed].mean()),
          supervised_field_mae=float((fields-supervised).abs().mean()),
          coincident_contour_disagreement=float(coincident_loss(nodes['up1_low_dw'],distance)),
          reference_profile_shape=float(profile['rgb_profile_shape']),reference_profile_contrast=float(profile['rgb_profile_contrast']),
          rank_violation=float(torch.cat([torch.diff(fields[:,lo:hi],dim=1) for lo,hi in GROUPS30],1).clamp_min(0).amax(1).mean()))
        records.append(record);im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);im.save(dest/(row['id']+'_integer720.png'))
        marked=np.array(im);marked[mask[0,0].numpy()]=[255,0,180]
        for board,pic in zip(boards,(Image.fromarray(target),im,Image.fromarray(marked),m.t.r.c.rgb_image(view[0]/128))):board[row['id']]=pic
        print('DIRECT_PAIR_QUALITY',row['id'],{k:round(record[k],6) for k in ('mae','bright_target_excess_fraction','ordinal_area_mae','mixed_ordinal_area_mae','rank_violation')},flush=True)
    make_board(boards,['target','actual bytes','bright diagnostic mask','field sum, diagnostic ONLY'],dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,constant1215_error=0,all_six720_all_node_error=0,
      bytes=20480,checkpoint_sha256=file_hash(a.checkpoint),visual_acceptance=False,
      supervised_field='native signed distance' if state.get('ordered_distance_gain') else 'native area',
      ordinal_area_metrics_are_diagnostic_only=bool(state.get('ordered_distance_gain')),
      limitation='Seen validation, brightness envelope is not semantic truth; no temporal acceptance from still images.'))

if __name__=='__main__':main()
