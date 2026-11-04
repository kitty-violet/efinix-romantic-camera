"""Isolate small-island survival in the existing ternary rounding layer.

Controlled reuse of spatial voting on the current three-state decoder. Only
up2_low_dw coefficients change; every result uses exported B1 bytes. This is
not an external filter, a new operator, or a minimum-area guarantee.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .seed_b1_ternary_regions import constant_check
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net,centre):
    kernel=net.weight['up2_low_dw'].new_tensor([[1,2,1],[2,centre,2],[1,2,1]])
    mult=round(131072*2/(127*float(kernel.sum())))
    net.weight['up2_low_dw'].zero_();net.bias['up2_low_dw'].zero_()
    net.weight['up2_low_dw'][:15,0]=kernel*mult/131072
    net.up2_low_dw_m.fill_(mult)
    return dict(centre=centre,kernel=kernel.tolist(),multiplier=mult)


@torch.inference_mode()
def island_check(net):
    records=[]
    for base in (0,127):
        for width in (1,2,3,8):
            z=torch.full((1,16,24,24),float(base))
            z[:,:15,10:10+width,10:10+width]=127-base
            result=fragment(net,z,'up1_high_dw')['up2_low_dw'][0,0]
            records.append(dict(background=base,width=width,changed_pixels=int((result!=(2 if base else 0)).sum()),
                centre_value=float(result[10,10])))
    return records


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);initial=copy.deepcopy(net.state_dict())
    rows=m.load_data(args.data)['validation'];images={};records=[]
    for centre in (20,4,2):
        name=f'centre{centre}';net.load_state_dict(initial);change=install(net,centre)
        constants=constant_check(net);islands=island_check(net)
        sub=dest/name;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw)
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
        saved.update(model=copy.deepcopy(net.state_dict()),ternary_spatial_vote=change,
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),added_training_steps=0,
            added_inference_operators=0,external_filter=False,visual_acceptance=False)
        torch.save(saved,sub/'initial.pt')
        mean={k:sum(r[k] for r in tests)/len(tests) for k in tests[0] if k!='id'}
        records.append(dict(name=name,change=change,constant=constants,island=islands,
            bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=tests,mean=mean))
        print('TERNARY_SPATIAL_VOTE',name,mean,islands[:2],flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for key,box,label in m.t.r.c.CROPS:
        board=Image.new('RGB',(1680,314),'#17202a');d=ImageDraw.Draw(board)
        pics=[('target',Image.open(by_id[key]['target']))]+[(f'centre{c}',images[f'centre{c}',key]) for c in (20,4,2)]
        for i,(name,im) in enumerate(pics):
            board.paste(im.crop(box),(420*i,34));d.text((420*i+4,5),label+' | '+name,font=font,fill='white')
        board.save(dest/(key+'_comparison.png'))
    write_json(dest/'review.json',dict(parent=args.checkpoint,records=records,added_training_steps=0,
        scope='Same ternary predictor; controlled existing-kernel vote. Not visual acceptance.'))


if __name__=='__main__':main()
