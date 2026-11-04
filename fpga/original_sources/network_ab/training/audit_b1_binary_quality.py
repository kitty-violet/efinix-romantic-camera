"""Joint integer quality audit: false bright spots, grid phase and colour layers.

No one number constitutes visual acceptance. Real white structures are retained;
the excess mask is a target-neighbourhood diagnostic, not semantic truth.
"""
import argparse
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_binary_layout import fields_for_layout
from .audit_b1_target_chain import region_means,adjacency,lab
from .audit_b1_white_rims_grid import measures
from .minimum_area_blocks import components
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);setup=state['binary_coverage720'];rows=m.load_data(a.data)['validation'];dest=Path(a.out);dest.mkdir(exist_ok=False)
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    z=torch.nn.functional.pad(fields_for_layout(torch.arange(1215)[:,None,None],setup)*96+16,(0,0,0,0,0,1))
    palette=fragment(net,z,'up1_low_pw')['rgb_head'][:,:,0,0].numpy()+128;records=[]
    for row in rows:
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target']);labels=m.load_region(row['region']).astype(np.int64)
        source_t=torch.from_numpy(source.transpose(2,0,1).copy())[None].float();target_t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float()
        nodes=net(source_t/128-1,all_nodes=True);other,_=actual(source_t-128,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        output=nodes['rgb_head']+128;bright,_=measures(output,target_t,source_t)
        means,count=region_means(target,labels);predicted,_=region_means(output[0].permute(1,2,0).numpy(),labels);aa,bb,length=adjacency(labels)
        tt,pp=lab(means),lab(predicted);wanted=tt[aa]-tt[bb];got=pp[aa]-pp[bb];strong=np.linalg.norm(wanted,axis=1)>=5
        den=max(1,length[strong].sum());idx=((means[:,None]-palette[None])**2).sum(2).argmin(1)
        collapsed=(np.linalg.norm(got,axis=1)<2)&strong;reversed=((wanted*got).sum(1)<0)&strong;pal_lost=(idx[aa]==idx[bb])&strong
        field=(nodes['up2_low_dw'][0,:15]>=64).long();offset=0;counts=[]
        for ww in setup['weights']:
            counts.append((field[offset:offset+len(ww)]*torch.tensor(ww)[:,None,None]).sum(0));offset+=len(ww)
        code=(81*counts[0]+9*counts[1]+counts[2]).numpy();groups=components(code);areas=np.bincount(groups.ravel())
        record=dict(id=row['id'],**bright,palette_lost_boundary_fraction=float(length[pal_lost].sum()/den),
            output_collapsed_boundary_fraction=float(length[collapsed].sum()/den),output_reversed_boundary_fraction=float(length[reversed].sum()/den),
            small_decision360_region_pixels=float(areas[areas<64].sum()/areas.sum()),decision360_regions=len(areas))
        records.append(record);print('BINARY_QUALITY',row['id'],{k:round(v,5) for k,v in record.items() if isinstance(v,float)},flush=True)
    scalars=[k for k,v in records[0].items() if isinstance(v,(int,float))]
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),actual_all_six720_node_error=0,byte_size=20480,
        records=records,mean={k:float(np.mean([r[k] for r in records])) for k in scalars},visual_acceptance=False,
        limitations='Previously seen validation sources. Region Lab separation and grid phase histograms are diagnostics, not visual or temporal acceptance.'))


if __name__=='__main__':main()
