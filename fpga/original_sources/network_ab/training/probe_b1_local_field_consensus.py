"""Existing DW performs a local vote of colour fields before hard decisions.

This changes genuine coefficient rows, not the displayed image. Uniform127
integer taps preserve a binary0/127 decision even for mixed neighbours. No
global connected-area guarantee follows from this local rule.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net,shape):
    kernel=torch.ones(3,3,device=net.weight['stem'].device)
    if shape=='cross':kernel[0,0]=kernel[0,2]=kernel[2,0]=kernel[2,2]=0
    assert shape in ('cross','box')
    size=int(kernel.sum());scale=131071/131072;key='up2_low_dw'
    net.weight[key].zero_();net.bias[key].zero_()
    net.weight[key][:15,0]=127*scale*kernel
    net.bias[key][:15]=-127*scale*int(63.5*size)
    getattr(net,key+'_m')[:,:15]=131071
    return kernel


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
        sub=dest/shape;sub.mkdir();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);v=nodes['up2_low_dw'][0,:15]
            assert ((v==0)|(v==127)).all()
            image=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128)
            image.save(sub/(row['id']+'_actual_B1.png'));images[row['id'],shape]=image
        if shape!='original':
            saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
            saved.update(model=copy.deepcopy(net.state_dict()),parent=a.warm,step=0,
                local_field_consensus=shape,manual_coefficient_change=True,requires_retraining=True)
            torch.save(saved,sub/'initial.pt')
        records.append(dict(shape=shape,package_sha256=__import__('hashlib').sha256(net.export()).hexdigest()))
        print('FIELD_CONSENSUS',shape,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by_id[key]['target'])]+[images[key,s] for s in ('original','cross','box')]
        for j,(im,title) in enumerate(zip(pics,('区域目标','原真实B1','十字判定·真实B1','方形判定·真实B1'))):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),records=records,
        added_operators=0,postfilter=False,binary_decisions_verified=True,visual_acceptance=False,
        global_minimum_area_guaranteed=False,manual_weight_probe=True))


if __name__=='__main__':main()
