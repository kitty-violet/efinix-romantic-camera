"""Controlled front-end swap: same late full-colour levels, exact rank filter."""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_direct_rgb_median import install as median
from .design_b1_joint_colour_levels import install as levels
from .design_b1_edge_stopping_diffusion import identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for mode in ('quadrants','rank_box','rank_gaussian'):
        net=copy.deepcopy(base);levels(net,2)
        if mode!='quadrants':
            front=copy.deepcopy(base);median(front)
            if mode=='rank_gaussian':
                identity_dw(front,'down1__dw',range(3),KERNEL);identity_dw(front,'down2__dw',range(3),KERNEL)
            for k in net.weight:
                if k=='up1_low_dw':break
                net.weight[k].copy_(front.weight[k]);net.bias[k].copy_(front.bias[k]);getattr(net,k+'_m').copy_(getattr(front,k+'_m'))
            net.package['residual_encoding']=copy.deepcopy(front.package['residual_encoding'])
        sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im;records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt');print('RANK_JOINT_PALETTE',mode,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('quadrants','rank_box','rank_gaussian')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
