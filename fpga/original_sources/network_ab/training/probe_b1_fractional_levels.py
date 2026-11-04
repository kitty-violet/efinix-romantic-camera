"""Actual-weight colour/edge probe: preserve half-levels at the final720 stage.

Unlike the old coverage4 experiment this starts from the reviewed halfgauss
field model and tests centre-weighted native kernels. No external image filter.
Manual coefficients are explicitly recorded; this is not new learned training.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_native_boundary_phase import install as install_field
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


@torch.no_grad()
def install_terminal(net,coverage,kernel):
    if kernel=='cross':k=net.weight['up2_high_dw'].new_tensor([[0,1,0],[1,12,1],[0,1,0]])
    elif kernel=='binomial':k=net.weight['up2_high_dw'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    elif kernel=='balanced':k=net.weight['up2_high_dw'].new_tensor([[1,2,1],[2,8,2],[1,2,1]])
    else:raise ValueError(kernel)
    for c,increment in enumerate((16,9,6)):
        mult=round(131072*coverage/(float(k.sum())*increment))
        net.up2_high_dw_m[:,c]=mult
        net.weight['up2_high_dw'][c,0]=k*mult/131072
        net.weight['rgb_head'][:,c]/=coverage


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);initial=copy.deepcopy(net.state_dict())
    rows=m.load_data(args.data)['validation'];images={};records=[]
    cases=[('current',5,1,None),('aa2',5,2,'balanced'),('aa4',5,4,'balanced'),
           ('span9aa2',9,2,'balanced'),('span13aa2',13,2,'balanced'),('span9cross',9,2,'cross')]
    patterns=((torch.arange(32768)[:,None]>>torch.arange(15)[None])&1).float()*127
    z=torch.zeros(32768,16,1,1);z[:,:15,0,0]=patterns
    reference=fragment(net,z,'up2_low_dw')['rgb_head'].clone()
    for name,span,coverage,kernel in cases:
        net.load_state_dict(initial)
        if span!=5:
            # Same halfgauss field, with a declared larger transition interval.
            shape=net.weight['up2_low_dw'].new_tensor([[1,2,1],[2,20,2],[1,2,1]])*6
            mult=round(131072*127/(float(shape.sum())*span));gain=float(shape.sum())*mult/131072
            net.weight['up2_low_dw'][:15,0]=shape*mult/131072
            net.bias['up2_low_dw'][:15]=-(127-span)/2*gain;net.up2_low_dw_m[:,:15]=mult
        if kernel:install_terminal(net,coverage,kernel)
        const=float((fragment(net,z,'up2_low_dw')['rgb_head']-reference).abs().max())
        assert const<=1,('constant colour regression',name,const)
        sub=dest/name;sub.mkdir();raw=net.export();(sub/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');tests=[]
        for row in rows:
            a=m.t.r.read(row['path']);x=torch.from_numpy(a.transpose(2,0,1).copy())[None].float()/128-1
            ours=net(x,all_nodes=True);ref,_=actual(x*128,True)
            err=max(float((v-ref[k]).abs().max()) for k,v in ours.items());assert err==0
            p=ref['rgb_head']/128;t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            loss,terms=joint_region_loss(p,t,labels)
            im=m.t.r.c.rgb_image(p[0]);im.save(sub/(row['id']+'.png'));images[name,row['id']]=im
            tests.append(dict(id=row['id'],all_node_error=err,score=float(loss),**{k:float(v) for k,v in terms.items()}))
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng','bounded_training_model')}
        saved.update(model=copy.deepcopy(net.state_dict()),manual_fractional_levels=dict(span=span,coverage=coverage,kernel=kernel),
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),added_training_steps=0,
            added_inference_operators=0,external_filter=False,visual_acceptance=False)
        torch.save(saved,sub/'candidate.pt')
        records.append(dict(name=name,constant_palette_error=const,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=tests))
        print('FRACTIONAL_LEVELS',name,{k:sum(r[k] for r in tests)/len(tests) for k in ('score','region_flatness','sharp_boundary')},flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for key,box,label in m.t.r.c.CROPS:
        board=Image.new('RGB',(1260,314*3),'#17202a');d=ImageDraw.Draw(board)
        pics=[('target',Image.open(by_id[key]['target']))]+[(n,images[n,key]) for n,_,_,_ in cases]
        for i,(name,im) in enumerate(pics):
            xx=i%3*420;yy=i//3*314;board.paste(im.crop(box),(xx,yy+34));d.text((xx+4,yy+5),label+' | '+name,font=font,fill='white')
        board.save(dest/(key+'_comparison.png'))
    write_json(dest/'review.json',dict(parent=args.checkpoint,records=records,added_training_steps=0,
        scope='Internal actual integer B1 decoder experiment. No accepted style claim or full-video test.'))


if __name__=='__main__':main()
