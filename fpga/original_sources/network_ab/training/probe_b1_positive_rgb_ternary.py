"""RGB count decoder without signed colour mixing; representation diagnostic.

Five ternary fields per R/G/B yield11^3 stable colours in the same15 fields.
Positive decoder coefficients provide a channel-envelope property for ordered
two-colour boundary inputs. This does NOT guarantee learned source predictions
or jointly convex RGB interpolation. Known target labels are injected here.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import joint_region_loss
from .audit_b1_white_rims_grid import measures
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


def states(codes):
    levels=(codes//121,codes//11%11,codes%11)
    return torch.cat([(v[:,None]-2*torch.arange(5,device=codes.device)[None,:,None,None]).clamp(0,2) for v in levels],1).float()


@torch.no_grad()
def install(net):
    keys=('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    net.up1_high_dw_m.fill_(65536);net.weight['up1_high_dw'][:15,0]=kernel*.5;net.bias['up1_high_dw'][:15]=-444.5
    vote=kernel.new_ones(3,3);vote[1,1]=20
    net.up2_low_dw_m.fill_(74);net.weight['up2_low_dw'][:15,0]=vote*74/131072
    net.up2_low_pw_m.fill_(131071)
    for c in range(3):net.weight['up2_low_pw'][c,c*5:(c+1)*5,0,0]=12*131071/131072
    net.up2_high_dw_m.fill_(8192);net.weight['up2_high_dw'][:3,0]=kernel/16
    net.rgb_head_m.fill_(16384);net.bias['rgb_head'].fill_(-128)
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=17/8


@torch.inference_mode()
def palette(net):
    code=torch.arange(1331)[:,None,None];z=F.pad(states(code),(0,0,0,0,0,1))
    return tuple(map(tuple,(fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]+128).tolist()))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);install(net);pk=palette(net);assert len(set(pk))==1331
    # Test all1331 constant states from the180p field interface, including
    # clipping and both upsampling stages, rather than just from count values.
    code=torch.arange(1331)[:,None,None];z=F.pad(net.round(63.5+(states(code)-1)*16),(0,0,0,0,0,1))
    rgb=fragment(net,z,'up1_low_pw')['rgb_head'][:,:,0,0]+128;assert torch.equal(rgb,torch.tensor(pk))
    g=torch.Generator().manual_seed(9);pair=torch.randint(0,1331,(128,2),generator=g);field=torch.where(torch.arange(32)[None,None,None,:]<16,states(pair[:,0,None,None]),states(pair[:,1,None,None])).expand(-1,-1,12,-1)
    q=fragment(net,F.pad(net.round(63.5+(field-1)*16),(0,0,0,0,0,1)),'up1_low_pw')['rgb_head']+128
    p=torch.tensor(pk);lo=torch.minimum(p[pair[:,0]],p[pair[:,1]])[:,:,None,None];hi=torch.maximum(p[pair[:,0]],p[pair[:,1]])[:,:,None,None]
    overshoot=torch.maximum(q-hi,lo-q).clamp_min(0);maximum=float(overshoot.max());assert maximum<=3,maximum
    # Rounding can depart by a few RGB units; do not state a zero-error bound.
    raw=net.export();assert len(raw)==20480;(dest/'template.bin').write_bytes(raw)
    saved=dict(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),qat=True,step=0,contract_sha256=state['contract_sha256'],positive_rgb_ternary=True,template_only=True,visual_acceptance=False,parent=a.checkpoint)
    torch.save(saved,dest/'template.pt');rows=m.load_data(a.data)['validation'];images={};records=[]
    for row in rows:
        def read(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()
        target,src=read(row['target']),read(row['path']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        ids=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        field=F.pad(net.round(F.avg_pool2d(63.5+(states(ids)-1)*16,4)),(0,0,0,0,0,1))
        p=fragment(net,field,'up1_low_pw')['rgb_head']/128;loss,v=joint_region_loss(p,target/128-1,labels);stats,_=measures(p*128+128,target,src)
        im=m.t.r.c.rgb_image(p[0]);images[row['id']]=im;im.save(dest/(row['id']+'_target_injected.png'));records.append(dict(id=row['id'],**{k:float(z) for k,z in v.items()},**stats))
        print('POSITIVE_RGB_TARGET_INJECTION',row['id'],float(loss),stats['bright_target_excess_fraction'],flush=True)
    board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);byid={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,im) in enumerate([('target',Image.open(byid[key]['target'])),('positive RGB target injection',images[key])]):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(records=records,constant_palette_size=1331,constant_palette_error=0,pair_channel_overshoot_rgb=maximum,
        trained=False,source_prediction=False,target_injection=True,extra_inference_ops=0,byte_size=20480,visual_acceptance=False,
        scope='Positive RGB mixes; channel-wise pair test is not proof of no learned halo or common RGB convex segment.'))


if __name__=='__main__':main()
