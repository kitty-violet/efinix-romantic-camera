"""Verify the6/5/4 monotone-major code through the fixed B1 decoder."""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .b1_binary_layout import RADIX_WEIGHTS,radix_fields,radix_tables
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_binary_confidence import set_gain
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import joint_region_loss
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,net=load(a.checkpoint);dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=copy.deepcopy(state['binary_coverage720']);setup.update(weights=[list(w) for w in RADIX_WEIGHTS],code_layout='radix_thermometer_654',gain=64,source_readout_not_fitted=True)
    set_gain(net,64);key='up2_low_pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(516)
    start=0
    for c,ww in enumerate(RADIX_WEIGHTS):
        net.weight[key][c,start:start+len(ww),0,0]=torch.tensor(ww)*16*516/131072;start+=len(ww)
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pal=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255)
    code=torch.arange(1215)[:,None,None];score=F.pad(radix_fields(code)*96+16,(0,0,0,0,0,1))
    assert torch.equal(fragment(net,score,'up1_low_pw')['rgb_head'][:,:,0,0]+128,pal)
    # Every binary field pattern, not merely the canonical paths.
    bits=((torch.arange(32768)[:,None]>>torch.arange(15)[None])&1).float();counts=[];start=0
    for ww in RADIX_WEIGHTS:
        counts.append((bits[:,start:start+len(ww)]*torch.tensor(ww)).sum(1));start+=len(ww)
    expected=(torch.stack(counts,1)@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255)
    for i in range(0,len(bits),1024):
        field=F.pad(bits[i:i+1024,:,None,None]*127,(0,0,0,0,0,1))
        assert torch.equal(fragment(net,field,'up2_low_dw')['rgb_head'][:,:,0,0]+128,expected[i:i+1024])
    raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    x=torch.randint(-128,128,(1,3,64,80),generator=torch.Generator().manual_seed(21)).float();nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
    saved=copy.deepcopy(state);saved.update(model=net.state_dict(),binary_coverage720=setup,source_readout_not_fitted=True,step=0,visual_acceptance=False)
    torch.save(saved,dest/'template_not_trained.pt');pk=tuple(map(tuple,pal.tolist()));rows={r['id']:r for r in m.load_data(a.data)['validation']}
    board=Image.new('RGB',(840,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);records=[]
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        row=rows[rid];x0,y0,_,_=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        t=torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        lab=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None];code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None]
        score=F.pad(F.avg_pool2d(radix_fields(code)*96+16,4),(0,0,0,0,0,1));assert torch.equal(score,score.round())
        p=fragment(net,score,'up1_low_pw')['rgb_head']/128;_,v=joint_region_loss(p[...,16:-16,16:-16],t[...,16:-16,16:-16],lab[...,16:-16,16:-16]);records.append(dict(crop=label,**{k:float(z) for k,z in v.items()}))
        for j,(name,v) in enumerate([('目标',t),('单调主色层·目标注入',p)]):
            board.paste(m.t.r.c.rgb_image(v[0]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(records=records,canonical_colour_error=0,
        all32768_binary_patterns_error=0,small_random_graph_error=0,byte_size=20480,
        code_tables=[t.int().tolist() for t in radix_tables()],source_prediction=False,visual_acceptance=False,
        checkpoint_sha256=file_hash(a.checkpoint)))
    print('RADIX_COVERAGE_PASS 1215colours/32768patterns/integer_nodes/20KiB',flush=True)


if __name__=='__main__':main()
