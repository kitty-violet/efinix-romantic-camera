"""Keep all14Y predicates until the360p decision; reserve2 rows for chroma.

Unlike the historical pair decoder this keeps one row per luminance threshold
and the sharp g11 ramp. It removes subtractive radix carries without reducing
the15 luminance levels. Co/Cg use continuous sums, not hard bits.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_ordered_distance_geometry import set_ramp
from .probe_b1_monotone_pair_decoder import make_board
from .audit_b1_colour_fringe_origin import fringe,two_colour_support
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net):
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_();net.up1_low_pw_m.fill_(65536)
    for i in range(14):net.weight[key][i,i,0,0]=1
    for row,lo,hi in ((14,14,22),(15,22,30)):
        net.weight[key][row,lo:hi,0,0]=1/16;net.up1_low_pw_m[:,row]=8192
    kernel=net.weight['up1_high_dw'][0].clone();mult=net.up1_high_dw_m[:,0].clone()
    net.weight['up1_high_dw'].copy_(kernel[None].expand_as(net.weight['up1_high_dw']))
    net.bias['up1_high_dw'].zero_();net.up1_high_dw_m.copy_(mult[:,None].expand_as(net.up1_high_dw_m))
    set_ramp(net,11)
    for row in (14,15):
        net.weight['up2_low_dw'][row].zero_();net.weight['up2_low_dw'][row,0,1,1]=1
        net.bias['up2_low_dw'][row]=0;net.up2_low_dw_m[:,row]=65536
    key='up2_low_pw';net.weight[key].zero_();net.bias[key].zero_();net.up2_low_pw_m.fill_(65536)
    mult=round(131072*8/127);net.up2_low_pw_m[:,0]=mult;net.weight[key][0,:14,0,0]=mult/131072
    net.weight[key][1,14,0,0]=1;net.weight[key][2,15,0,0]=1


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);net=copy.deepcopy(base);install(net)
    out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data)
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    palette=fragment(base,z,'up1_low_dw')['rgb_head'];assert torch.equal(fragment(net,z,'up1_low_dw')['rgb_head'],palette)
    # Every legal two-level scalar mixture is monotone in decoded luminance.
    curve=[]
    for low in range(15):
        for high in range(low+1,15):
            ca=81*low+40;cb=81*high+40;bits=code_bits(torch.tensor([ca,cb])[:,None,None])
            t=torch.arange(128)[:,None,None,None]/127
            z=F.pad(bits[:1]*127*(1-t)+bits[1:]*127*t,(0,0,0,0,0,2))
            y=fragment(net,net.round(z),'up1_low_dw')['up2_high_dw'][:,0,0,0]
            assert (torch.diff(y)>=0).all() and y.min()==low*8 and y.max()==high*8
            curve.append(float(torch.diff(y).max()))
    raw=net.export();assert len(raw)==20480
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    ss={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
    ss.update(model=copy.deepcopy(net.state_dict()),direct_luminance_planes=True,radix_luma_continuous_chroma=True,
        luminance_layers=15,nonnegative_luma_decoder=True,extra_inference_operations=0,visual_acceptance=False)
    torch.save(ss,out/'initial.pt');(out/'coefficients-padded.bin').write_bytes(raw)
    pk=tuple(map(tuple,(palette[:,:,0,0]+128).tolist()));pal=np.array(pk,np.float32)
    boards=[{}, {}, {}];records=[]
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        old=base(x,all_nodes=True);nodes=net(x,all_nodes=True);other,_=actual(net.round(x*128),True)
        assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        target=m.t.r.read(row['target']).astype(np.float32);boards[0][row['id']]=m.t.r.c.Image.fromarray(target.astype(np.uint8)) if hasattr(m.t.r.c,'Image') else None
        from PIL import Image
        boards[0][row['id']]=Image.fromarray(target.astype(np.uint8))
        codes=region_codes(row['target'],row['region'],pk).astype(np.int64);ca,cb,valid=two_colour_support(codes,pal)
        for j,(name,nn) in enumerate((('radixY',old),('direct14Y',nodes)),1):
            p=nn['rgb_head'][0].permute(1,2,0).numpy()+128;im=Image.fromarray(p.astype(np.uint8));boards[j][row['id']]=im
            im.save(out/f'{name}_{row["id"]}.png')
            records.append(dict(id=row['id'],mode=name,mae=float(np.abs(p-target).mean()),fringe=fringe(p,ca,cb,valid)))
        print('DIRECT_LUMA_PLANES',records[-2:],flush=True)
    make_board(boards,['目标','6位亮度编码+连续色度','14独立亮度阈值+连续色度'],out/'comparison.png')
    write_json(out/'review.json',dict(records=records,all_nodes_integer_error=0,constant_palette_error=0,
        monotone_mixture_pairs=len(curve),bytes=20480,extra_inference_operations=0,visual_acceptance=False,
        source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
