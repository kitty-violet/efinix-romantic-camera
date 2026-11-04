"""Test one existing720p channel for dark-side transition shoulders.

Unlike the earlier four-Sobel ink test, only negative luminance curvature is
subtracted. Constant colours remain exact and no pixel can become brighter.
This may still create dark halos/grid edges, so image review is mandatory.
"""
import argparse,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_monotone_pair_decoder import make_board
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,strength):
    assert net.weight['up2_low_pw'][3:].abs().max()==0 and net.weight['rgb_head'][:,3:].abs().max()==0
    key='up2_low_pw';net.weight[key][3]=net.weight[key][0];net.bias[key][3]=net.bias[key][0];getattr(net,key+'_m')[:,3]=getattr(net,key+'_m')[:,0]
    key='up2_high_dw';getattr(net,key+'_m')[:,3]=4096
    net.weight[key][3,0]=net.weight[key].new_tensor([[1,2,1],[2,-12,2],[1,2,1]])/16
    net.bias[key][3]=-.5;net.weight['rgb_head'][:,3,0,0]=-strength

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2));palette=fragment(base,z,'up1_low_dw')['rgb_head'];rows=m.load_data(a.data)['validation'];boards=[];original={}
    for strength in (0.,1.,2.,3.):
        net=copy.deepcopy(base);install(net,strength);assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'],palette)
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={}
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            out=nodes['rgb_head']
            if strength==0:original[row['id']]=out.clone()
            else:assert (out<=original[row['id']]).all()
            im=m.t.r.c.rgb_image(out[0]/128);ims[row['id']]=im;im.save(dest/f's{strength}_{row["id"]}.png')
        boards.append(ims);saved=copy.deepcopy(state);saved.update(model=net.state_dict(),dark_shoulder_existing_channel=strength,visual_acceptance=False);torch.save(saved,dest/f's{strength}.pt')
        print('DARK_SHOULDER_PASS',strength,'1215constant/no_new_brightness/6x720nodes/20KiB',flush=True)
    make_board(boards,['unchanged','dark shoulder1','dark shoulder2','dark shoulder3'],dest/'source_crops.png')
    write_json(dest/'review.json',dict(parent_sha256=file_hash(a.checkpoint),constant1215_error=0,all_six720_node_error=0,
      extra_operators=0,bytes=20480,no_pixel_brightened=True,visual_acceptance=False))

if __name__=='__main__':main()
