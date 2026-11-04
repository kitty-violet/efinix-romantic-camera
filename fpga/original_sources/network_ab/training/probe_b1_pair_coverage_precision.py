"""Keep pure palette colours but refine subpixel coverage in existing DW/PW.

Increasing a pair's0..2 range to0..4/0..8 and inversely scaling the next PW
retains all1215 pure colours. Intermediate boundary coverage gets more legal
values. This is not a claim that soft source predictions become sharp.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_monotone_pair_decoder import GROUPS15,make_board
from .build_b1_thirty_binary_regions import code_bits
from .audit_b1_binary_boundary_phase import contour_metrics
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,maximum):
    assert maximum in (2,4,8)
    mult=1032*maximum;key='up2_low_dw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(mult)
    net.weight[key][:15,0,1,1]=mult/131072
    key='up2_low_pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    for c,(lo,hi) in enumerate(GROUPS15):net.weight[key][c,lo:hi,0,0]=(16//maximum)*131071/131072

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);setup=state.get('ordinal_pair_setup') or state['binary_coverage720'];dest=Path(a.out);dest.mkdir(exist_ok=False)
    matrix=torch.tensor(setup['matrix']).float();bias=torch.tensor(setup['bias']).float();levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();palette=(levels@matrix.T+bias).clamp(-128,127)
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij');boards=[];records=[]
    for maximum in (2,4,8):
        net=copy.deepcopy(base);install(net,maximum)
        z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2));assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'][:,:,0,0],palette)
        for ca,cb in (((5,4,4),(9,4,4)),((5,3,4),(6,6,3)),((2,4,5),(11,4,3))):
            c0=(matrix@torch.tensor(ca).float()+bias).clamp(-128,127);c1=(matrix@torch.tensor(cb).float()+bias).clamp(-128,127)
            for slope in (.2,.5,1.):
                for phase in range(4):
                    truth=64+slope*(torch.arange(96)-48)+phase-.5
                    code=torch.where(xx>truth[:,None],cb[0]*81+cb[1]*9+cb[2],ca[0]*81+ca[1]*9+ca[2])[None]
                    z=F.pad(net.round(F.avg_pool2d(code_bits(code)*127,4)),(0,0,0,0,0,2))
                    rgb=fragment(net,z,'up1_low_dw')['rgb_head'][0]
                    records.append(dict(maximum=maximum,**contour_metrics(rgb,c0,c1,truth)))
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={}
        for row in m.load_data(a.data)['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);ims[row['id']]=im;im.save(dest/f'q{maximum}_{row["id"]}.png')
        boards.append(ims);saved=copy.deepcopy(state);saved.update(model=net.state_dict(),ordinal_pair_coverage_maximum=maximum,visual_acceptance=False);torch.save(saved,dest/f'q{maximum}.pt')
        subset=[r for r in records if r['maximum']==maximum];print('PAIR_COVERAGE_PRECISION_PASS',maximum,{k:float(np.mean([r[k] for r in subset])) for k in ('contour_rms_px','boundary_width_px','backtrack')},flush=True)
    make_board(boards,['pair0..2','pair0..4','pair0..8'],dest/'source_crops.png')
    write_json(dest/'review.json',dict(records=records,constant1215_error=0,source_replay_error=0,bytes=20480,
      extra_operators=0,parent_sha256=file_hash(a.checkpoint),visual_acceptance=False))

if __name__=='__main__':main()
