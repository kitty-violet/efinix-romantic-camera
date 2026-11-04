"""Existing-channel bright-side curvature correction, not a postfilter.

The old dark-shoulder probe used opposite curvature and darkened the dark side.
Here test the bright side at720 and360, preserve all constant palette entries,
bound correction and inspect whether removing a pale shoulder also softens it.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_monotone_pair_decoder import make_board
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_white_rims_grid import measures
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,scale,strength):
    assert net.weight['up2_low_pw'][3:].abs().max()==0 and net.weight['rgb_head'][:,3:].abs().max()==0
    kernel=net.weight['rgb_head'].new_tensor([[-1,-2,-1],[-2,12,-2],[-1,-2,-1]])/16
    if scale==720:
        net.weight['up2_low_pw'][3]=net.weight['up2_low_pw'][0];net.bias['up2_low_pw'][3]=net.bias['up2_low_pw'][0];net.up2_low_pw_m[:,3]=net.up2_low_pw_m[:,0]
        net.weight['up2_high_dw'][3,0]=kernel;net.bias['up2_high_dw'][3]=-1;net.up2_high_dw_m[:,3]=4096
    else:
        assert net.weight['up1_low_pw'][15].abs().max()==0
        net.up1_low_pw_m[:,15]=8256;net.weight['up1_low_pw'][15,:14,0,0]=8256/131072;net.bias['up1_low_pw'][15]=0
        net.weight['up1_high_dw'][15,0]=kernel;net.bias['up1_high_dw'][15]=0;net.up1_high_dw_m[:,15]=4096
        smooth=kernel.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
        net.weight['up2_low_dw'][15,0]=smooth;net.bias['up2_low_dw'][15]=-1;net.up2_low_dw_m[:,15]=8192
        net.weight['up2_low_pw'][3].zero_();net.weight['up2_low_pw'][3,15,0,0]=1;net.bias['up2_low_pw'][3]=0;net.up2_low_pw_m[:,3]=65536
        net.weight['up2_high_dw'][3,0]=smooth;net.bias['up2_high_dw'][3]=0;net.up2_high_dw_m[:,3]=8192
    net.weight['rgb_head'][:,3,0,0]=-strength


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2));palette=fragment(base,z,'up1_low_dw')['rgb_head'];rows=m.load_data(a.data)['validation'];boards=[];original={};records=[]
    variants=(('base',720,0),('bright720_1',720,1.),('bright360_1',360,1.),('bright360_2',360,2.))
    for name,scale,strength in variants:
        net=copy.deepcopy(base)
        if strength:install(net,scale,strength)
        assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'],palette)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()-128
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            out=nodes['rgb_head']
            if strength==0:original[row['id']]=out.clone()
            else:assert (out<=original[row['id']]).all()
            profile,v=reference_boundary_profile(out/128,t/128,labels);bright,_=measures(out+128,t+128,x+128)
            rec=dict(id=row['id'],variant=name,mae=float((out-t).abs().mean()),profile=float(profile),maximum_darken=float((original[row['id']]-out).max()),**bright)
            records.append(rec);im=m.t.r.c.rgb_image(out[0]/128);ims[row['id']]=im;im.save(dest/f'{name}_{row["id"]}.png')
        boards.append(ims);saved=copy.deepcopy(state);saved.update(model=net.state_dict(),bright_shoulder_scale=scale,bright_shoulder_strength=strength,manual_weight_edit=True,added_training_steps=0,visual_acceptance=False);torch.save(saved,dest/f'{name}.pt')
        print('BRIGHT_SHOULDER_PASS',name,'1215constant/no_new_brightness/6x720nodes/20KiB',[(r['id'],round(r['profile'],4)) for r in records if r['variant']==name],flush=True)
    make_board(boards,[r[0] for r in variants],dest/'comparison.png')
    write_json(dest/'review.json',dict(records=records,parent_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),constant1215_error=0,all_six720_node_error=0,extra_operators=0,bytes=20480,no_pixel_brightened=True,visual_acceptance=False))


if __name__=='__main__':main()
