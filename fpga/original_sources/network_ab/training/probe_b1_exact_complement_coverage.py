"""Remove integer complement bias in the compiled colour-field decoder.

Use0/127 scores and an odd integer slope centred at63.5. Complementary fields
then remain complementary through the positive odd-sum spatial stencil and
the clipping ramp. Original Q17 B1 operations and20KiB layout are unchanged.
"""
import argparse,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_binary_layout import radix_tables,fields_for_layout
from .build_b1_thirty_binary_regions import code_bits,GROUPS30
from .audit_b1_common_coordinate_boundaries import hinges
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,gain=7):
    assert gain%2==1 and gain<=127
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071);offset=0
    for table,(a,b) in zip(radix_tables(net.weight[key].device),GROUPS30):
        diff=torch.diff(table,dim=0).T;net.weight[key][offset:offset+len(diff),a:b,0,0]=diff*131071/131072;offset+=len(diff)
    key='up2_low_dw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    net.weight[key][:15,0,1,1]=gain*131071/131072
    net.bias[key][:15]=(127*(1-gain)//2)*131071/131072


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gain',type=int,default=7);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False);setup=copy.deepcopy(state['binary_coverage720']);install(net,a.gain)
    q=torch.arange(1906).float();norm=lambda v:net.round(v.double()*8738/131072)
    assert torch.equal(norm(q)+norm(1905-q),torch.full_like(norm(q),127))
    v=torch.arange(128).float();ramp=lambda z:net.round((z*a.gain+127*(1-a.gain)//2).double()*131071/131072).clamp(0,127)
    assert torch.equal(ramp(v)+ramp(127-v),torch.full_like(ramp(v),127))
    code=torch.arange(1215)[:,None,None];h=F.pad(code_bits(code)*127,(0,0,0,0,0,2));nodes=fragment(net,h,'res3_conv1.pw')
    assert torch.equal(nodes['up1_low_pw'][:,:15],fields_for_layout(code,setup)*127)
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();pal=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127)
    assert torch.equal(nodes['rgb_head'][:,:,0,0],pal)
    scans=[]
    for axis,maximum in enumerate((14,8,8)):
        v=torch.linspace(0,maximum,maximum*128+1);c=torch.tensor([7.,4.,4.])[None,:,None,None].repeat(len(v),1,1,1);c[:,axis,0,0]=v
        rgb=fragment(net,net.round(hinges(c)),'res3_conv1.pw')['rgb_head'][:,:,0,0]
        direction=torch.tensor(setup['matrix']).float()[:,axis];position=(rgb*direction).sum(1)/direction.square().sum()
        minimum=float(torch.diff(position).min());assert minimum>=-1e-5,(axis,minimum)
        scans.append(dict(axis=axis,minimum_coordinate_step=minimum,negative_steps=0))
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    for row in m.load_data(a.data)['validation']:
        src=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
        nodes=net(src/128,all_nodes=True);other,_=actual(src,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        m.t.r.c.rgb_image(nodes['rgb_head'][0]/128).save(dest/(row['id']+'_output.png'))
    setup.update(gain=a.gain,score_center=63.5,score_endpoints=[0,127],exact_integer_complement=True)
    saved=copy.deepcopy(state);saved.update(model=net.state_dict(),binary_coverage720=setup,step=0,visual_acceptance=False)
    torch.save(saved,dest/'candidate.pt');write_json(dest/'review.json',dict(uniform_scan=scans,odd_stencil_complement_error=0,
        ramp_complement_error=0,constant_palette_error=0,actual_all_six720_node_error=0,byte_size=20480,
        visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))
    print('EXACT_COMPLEMENT_PASS 1906stencil/128ramp/1215colours/monotone_scans/6x720_integer',flush=True)


if __name__=='__main__':main()
