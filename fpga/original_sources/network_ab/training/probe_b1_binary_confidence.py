"""Counterfactual: narrow the binary transition without changing colour levels.

Measure actual source outputs before retraining. A narrow ramp alone can expose
wrong/noisy decisions; this diagnostic does not claim a finished style model.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_symmetric_binary_coverage import regression
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_boundary_profile import boundary_profile
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.no_grad()
def set_gain(net,gain):
    key='up2_low_dw';net.weight[key].zero_();net.bias[key].zero_()
    if gain<64:
        getattr(net,key+'_m').fill_(65536);net.weight[key][:15,0,1,1]=gain
        net.bias[key][:15]=63.5-64*gain
    else:
        assert gain==64
        s=131071/131072;getattr(net,key+'_m').fill_(131071)
        net.weight[key][:15,0,1,1]=64*s;net.bias[key][:15]=(64-64*64)*s


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False);state,base=load(a.checkpoint)
    nets={};actuals={}
    for gain in (8,16,64):
        net=copy.deepcopy(base);set_gain(net,gain);setup=copy.deepcopy(state['binary_coverage720']);setup['gain']=gain
        regression(net,setup);raw=net.export();assert len(raw)==20480
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),binary_coverage720=setup,visual_acceptance=False,step=0)
        torch.save(saved,dest/f'gain{gain}_needs_retraining.pt');nets[gain]=net
        actuals[gain]=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    rows={r['id']:r for r in m.load_data(a.data)['validation']};records=[]
    board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        row=rows[rid];x0,y0,_,_=box;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+280,x0:x0+420]
        def read(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy())[None].float()/128-1
        x,t=read(row['path']),read(row['target']);lab=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        pictures=[('目标',t)];records.append(dict(id=rid,crop=label,variants={}))
        for gain,net in nets.items():
            nodes=net(x,all_nodes=True);other,_=actuals[gain](x*128,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            p=nodes['rgb_head']/128;_,v=joint_region_loss(p[...,40:-40,40:-40],t[...,40:-40,40:-40],lab[...,40:-40,40:-40]);_,prof=boundary_profile(p,t,lab)
            field=nodes['up2_low_dw'][:,:15,...,20:-20,20:-20];active=(field>0)&(field<127)
            records[-1]['variants'][str(gain)]={**{k:float(z) for k,z in (v|prof).items()},'any_active':float(active.any(1).float().mean()),'active_fields':float(active.float().mean())}
            pictures.append((f'源预测·坡度{gain}',p))
        for j,(name,p) in enumerate(pictures):
            board.paste(m.t.r.c.rgb_image(p[0]),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
        print('BINARY_CONFIDENCE',label,[(g,v['any_active'],v['region_flatness']) for g,v in records[-1]['variants'].items()],flush=True)
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,source_prediction=True,
        actual_crop_nodes_error=0,constant_palette_error=0,needs_retraining=True,visual_acceptance=False,
        checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
