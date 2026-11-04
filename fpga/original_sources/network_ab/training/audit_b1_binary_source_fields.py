"""Source classification and coverage trade-off with exact integer replay."""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_binary_confidence import set_gain
from .probe_b1_decoder_coverage import fragment
from .b1_binary_layout import fields_for_layout
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_boundary_profile import boundary_profile
from .train_b1_binary_core_confidence import core_confidence,score360
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False);state,base=load(a.checkpoint);setup=state['binary_coverage720']
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255);pk=tuple(map(tuple,palette.tolist()))
    nets={};actuals={}
    for gain in (64,16,8,4):
        net=copy.deepcopy(base);set_gain(net,gain)
        score=F.pad(fields_for_layout(torch.arange(1215)[:,None,None],setup)*96+16,(0,0,0,0,0,1))
        assert torch.equal(fragment(net,score,'up1_low_pw')['rgb_head'][:,:,0,0]+128,palette)
        raw=net.export();assert len(raw)==20480;nets[gain]=net;actuals[gain]=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        saved=copy.deepcopy(state);saved['binary_coverage720']['gain']=gain;saved['model']=net.state_dict();saved['visual_acceptance']=False
        torch.save(saved,dest/f'gain{gain}_candidate.pt')
    rows={r['id']:r for r in m.load_data(a.data)['validation']};records=[]
    board=Image.new('RGB',(5*420,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        row=rows[rid];x0,y0,_,_=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        def read(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy())[None].float()/128-1
        x,t=read(row['path']),read(row['target']);lab=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64))[None];truth=F.avg_pool2d(fields_for_layout(code,setup),2)
        core=(F.max_pool2d(truth,5,1,2)==0)|(-F.max_pool2d(-truth,5,1,2)==1)
        core[...,:4,:]=False;core[...,-4:,:]=False;core[...,:,:4]=False;core[...,:,-4:]=False
        pictures=[('目标',t)];record=dict(id=rid,crop=label,variants={})
        for gain,net in nets.items():
            nodes=net(x,all_nodes=True);other,_=actuals[gain](x*128,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            p=nodes['rgb_head']/128;_,v=joint_region_loss(p[...,40:-40,40:-40],t[...,40:-40,40:-40],lab[...,40:-40,40:-40]);_,prof=boundary_profile(p,t,lab)
            score=score360(net,nodes);_,conf=core_confidence(score,truth);wrong=((score>64)!=(truth>.5))&core
            field=nodes['up2_low_dw'][:,:15];active=(field>0)&(field<127)
            # Positive-kernel rank preservation can be assessed per main group.
            rank=[];offset=0
            for ww in setup['weights']:
                major=score[:,offset+2:offset+len(ww)];rank.append(float((torch.diff(major,dim=1)>0).any(1).float().mean()));offset+=len(ww)
            record['variants'][str(gain)]={**{k:float(z) for k,z in (v|prof|conf).items()},
                'wrong_by_field':(wrong.sum((0,2,3))/core.sum((0,2,3)).clamp_min(1)).tolist(),
                'core_active_fields':float((active&core).sum()/core.sum()),'any_active':float(active.any(1).float().mean()),
                'major_rank_violation':rank}
            pictures.append((f'源预测·坡度{gain}',p))
        for j,(name,p) in enumerate(pictures):
            board.paste(m.t.r.c.rgb_image(p[0]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
        records.append(record);print('BINARY_SOURCE_FIELDS',label,[(k,v['core_active_fields'],v['region_flatness']) for k,v in record['variants'].items()],flush=True)
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,source_prediction=True,actual_crop_nodes_error=0,
        constant_palette_error=0,byte_size=20480,visual_acceptance=False,checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
