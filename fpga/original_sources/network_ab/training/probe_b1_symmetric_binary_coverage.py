"""Test symmetric binary coverage without colour-dependent ternary ramps.

Existing15 fields encode weighted binary counts (15/9/9 colours). Scores are
centred at64; an odd-sum spatial stencil avoids complementary half-tie drift.
The existing DW clips a narrow continuous ramp, not an independent hard bit.
Known-target injection is a decoder diagnostic only, never source prediction.
"""
import argparse,copy,itertools
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_boundary_profile import boundary_profile
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash

WEIGHTS=((1,2,3,4,4),(1,1,2,2,2),(1,1,2,2,2))


def tables(device='cpu'):
    bits=((torch.arange(32)[:,None]>>torch.arange(5)[None])&1).float()
    result=[]
    for weights in WEIGHTS:
        counts=(bits*torch.tensor(weights)).sum(1).long();layers=[torch.where(counts==i)[0].tolist() for i in range(sum(weights)+1)]
        costs={0:(0.,[0])}
        for layer in layers[1:]:
            next_cost={}
            for j in layer:
                options=[(cost+float((bits[i]-bits[j]).abs().sum()),path+[j]) for i,(cost,path) in costs.items()]
                next_cost[j]=min(options,key=lambda q:q[0])
            costs=next_cost
        _,path=min(costs.values(),key=lambda q:q[0]);result.append(bits[path].to(device))
    return result


def code_fields(code):
    return torch.cat([tab[level] for tab,level in zip(tables(code.device),(code//81,code//9%9,code%9))],-1).permute(0,3,1,2)


@torch.no_grad()
def install(net,setup,gain=16):
    device=net.weight['rgb_head'].device;matrix=torch.tensor(setup['matrix'],device=device)
    for key in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'):
        net.weight[key].zero_();net.bias[key].zero_()
    # Sum15 is deliberate: for integer dot products, complementary scores
    # around64 cannot both land on a half tie after normalization.
    kernel=matrix.new_tensor([[1,2,1],[2,3,2],[1,2,1]])
    net.up1_high_dw_m.fill_(8738);net.weight['up1_high_dw'][:15,0]=kernel*(8738/131072)
    net.up2_low_dw_m.fill_(65536)
    net.weight['up2_low_dw'][:15,0,1,1]=gain
    net.bias['up2_low_dw'][:15]=63.5-64*gain
    net.up2_low_pw_m.fill_(516)
    for c,ww in enumerate(WEIGHTS):net.weight['up2_low_pw'][c,c*5:c*5+5,0,0]=torch.tensor(ww,device=device)*16*516/131072
    net.up2_high_dw_m.fill_(8192)
    net.weight['up2_high_dw'][:3,0]=matrix.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.rgb_head_m.fill_(16384);net.weight['rgb_head'][:,:3,0,0]=matrix/8
    net.bias['rgb_head'].copy_(matrix.new_tensor(setup['bias']))
    # No claim that the inherited encoder predicts these new binary scores.
    return dict(matrix=setup['matrix'],bias=setup['bias'],weights=[list(w) for w in WEIGHTS],
                gain=gain,score_center=64,score_endpoints=[16,112],odd_stencil_sum=15,
                count_scale=8,source_readout_not_fitted=True)


@torch.inference_mode()
def regression(net,setup):
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pal=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255)
    code=torch.arange(1215)[:,None,None];score=F.pad(code_fields(code)*96+16,(0,0,0,0,0,1))
    got=fragment(net,score,'up1_low_pw')['rgb_head'][:,:,0,0]+128;assert torch.equal(got,pal),float((got-pal).abs().max())
    # Exhaustive integer stencil-dot and clipped ramp complement tests.
    dot=torch.arange(0,128*15+1).double();norm=net.round(dot*8738/131072)
    assert torch.equal(norm+norm.flip(0),torch.full_like(norm,128))
    score=torch.arange(1,128).float()
    injected=torch.zeros(127,16,1,1);injected[:,:15,0,0]=score[:,None]
    v=fragment(net,injected,'up1_high_dw')['up2_low_dw'][:,0,0,0]
    expected=net.round(63.5+setup['gain']*(score-64)).clamp(0,127)
    assert torch.equal(v,expected)
    assert (v+v.flip(0)-127).abs().max()<=1
    return pal


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False);state,base=load(a.checkpoint)
    setup0=state['aggregate720'];rows={r['id']:r for r in m.load_data(a.data)['validation']};nets={};pal=None
    for gain in (8,16,32):
        net=copy.deepcopy(base);setup=install(net,setup0,gain);pal=regression(net,setup);raw=net.export();assert len(raw)==20480
        real=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');x=torch.randint(-128,128,(1,3,64,80),generator=torch.Generator().manual_seed(12)).float()
        full=net(x/128,all_nodes=True);other,_=real(x,True);assert all(torch.equal(v,other[k]) for k,v in full.items())
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),binary_coverage720=setup,aggregate720=False,
            ternary_packed720=False,step=0,visual_acceptance=False,source_readout_not_fitted=True)
        torch.save(saved,dest/f'template_gain{gain}_not_trained.pt');nets[gain]=net
    pk=tuple(map(tuple,pal.tolist()));records=[];board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        row=rows[rid];x0,y0,_,_=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        t=torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        lab=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        score=F.pad(F.avg_pool2d(code_fields(code)*96+16,4),(0,0,0,0,0,1));assert torch.equal(score,score.round())
        pictures=[('目标',t)];records.append(dict(id=rid,crop=label,variants={}))
        for gain,net in nets.items():
            p=fragment(net,score,'up1_low_pw')['rgb_head']/128
            _,v=joint_region_loss(p[...,16:-16,16:-16],t[...,16:-16,16:-16],lab[...,16:-16,16:-16]);_,prof=boundary_profile(p,t,lab)
            records[-1]['variants'][str(gain)]={k:float(q) for k,q in (v|prof).items()};pictures.append((f'目标注入·对称二态{gain}',p))
        for j,(name,p) in enumerate(pictures):
            board.paste(m.t.r.c.rgb_image(p[0]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
        print('SYMMETRIC_BINARY',label,[(k,v['region_flatness'],v['rgb_profile_shape']) for k,v in records[-1]['variants'].items()],flush=True)
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(records=records,constant_palette_error=0,
        complementary_stencil_exhaustive_error=0,complementary_clip_max_error=1,small_graph_node_error=0,
        byte_size=20480,new_inference_ops=0,source_prediction=False,visual_acceptance=False,
        checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
