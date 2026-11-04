"""Two successive decisions inside existing B1 DW rows; no external filter.

First steepen the current360p Gaussian field into a near-binary decision,
then vote those decisions in the next existing DW. This differs from averaging
analogue fields before their first threshold. The first decision can retain
intermediate integer codes; the final vote is checked strictly binary.
"""
import argparse
import copy
import hashlib
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_local_field_consensus import install as vote
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net,shape,gain=256):
    key='up1_high_dw';mult=getattr(net,key+'_m');old=mult.clone()
    new=old[:,:15]*gain;assert new.max()<=131071
    net.weight[key][:15]*=gain
    net.bias[key][:15]=gain*net.bias[key][:15]+(1-gain)*63.5
    mult[:,:15]=new
    vote(net,shape)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    initial=copy.deepcopy(net.state_dict());rows=m.load_data()['validation'];images={};records=[]
    for shape in ('original','cross','box'):
        net.load_state_dict(initial)
        if shape!='original':install(net,shape)
        sub=dest/shape;sub.mkdir();raw=net.export();(sub/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');details=[]
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);reference=net(x/128,all_nodes=True)
            error=max(float((nodes[k]-reference[k]).abs().max()) for k in nodes);assert error==0
            v=nodes['up2_low_dw'][0,:15];assert ((v==0)|(v==127)).all()
            first=nodes['up1_high_dw'][0,:15]
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);images[row['id'],shape]=im
            im.save(sub/(row['id']+'_actual_B1.png'))
            details.append(dict(id=row['id'],all_node_error=error,
                first_decision_intermediate_fraction=float(((first!=0)&(first!=127)).float().mean())))
        if shape!='original':
            saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
            saved.update(model=copy.deepcopy(net.state_dict()),parent=a.warm,step=0,
                decision_then_vote=shape,manual_coefficient_change=True,
                training_proxy_requires_update=True,minimum_area_inference_guarantee=False)
            torch.save(saved,sub/'initial.pt')
        records.append(dict(shape=shape,package_sha256=hashlib.sha256(raw).hexdigest(),records=details))
        print('DECISION_THEN_VOTE',shape,'six native frames all nodes=0',flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by_id[key]['target'])]+[images[key,s] for s in ('original','cross','box')]
        for j,(im,title) in enumerate(zip(pics,('区域目标','原真实B1','先判定再十字投票·B1','先判定再方形投票·B1'))):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),records=records,
        added_operators=0,postfilter=False,final_binary_decisions_verified=True,visual_acceptance=False,
        global_minimum_area_guaranteed=False,actual_board=False,manual_weight_probe=True))


if __name__=='__main__':main()
