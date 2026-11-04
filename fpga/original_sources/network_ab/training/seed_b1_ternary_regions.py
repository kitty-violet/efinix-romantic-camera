"""Use three stable states per existing region field, with native edge coverage.

Old fields choose only0/127, restricting interiors to8x5x5 colours. Here the
existing up1_high_dw clips the amplified field and up2_low_dw rounds its local
mean to0/1/2. The same grouped PW sums them; native720 keeps half-step edge
coverage. Existing B1 layers, INT8/Q17 arithmetic, parameter count and bytes.
Manual initialization, not proof that the source predictor learned the target.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import GROUPS
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net,gain):
    # Change only the existing affine field and fixed decoder coefficients.
    net.weight['up1_high_dw'][:15]*=gain
    net.bias['up1_high_dw'][:15]=gain*(net.bias['up1_high_dw'][:15]-63.5)+63.5
    key='up1_high_dw';w=net.weight[key]
    mult=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/112).ceil().clamp(1,131071)
    getattr(net,key+'_m').copy_(mult.double()[None,:,None,None])
    # Average then integer round: 0<=input<=127 ensures exactly three states.
    kernel=w.new_tensor([[1,2,1],[2,20,2],[1,2,1]])
    den=float(kernel.sum())*127;mult=round(131072*2/den)
    net.weight['up2_low_dw'].zero_();net.bias['up2_low_dw'].zero_()
    net.weight['up2_low_dw'][:15,0]=kernel*mult/131072;net.up2_low_dw_m.fill_(mult)
    net.weight['up2_low_pw'].zero_();net.bias['up2_low_pw'].zero_();net.up2_low_pw_m.fill_(65536)
    for c,((a,b),inc) in enumerate(zip(GROUPS,(16,9,6))):net.weight['up2_low_pw'][c,a:b,0,0]=inc/2
    kernel=w.new_tensor([[1,2,1],[2,8,2],[1,2,1]])
    net.weight['up2_high_dw'].zero_();net.bias['up2_high_dw'].zero_()
    for c,inc in enumerate((16,9,6)):
        mm=round(131072*4/(float(kernel.sum())*inc));net.up2_high_dw_m[:,c]=mm
        net.weight['up2_high_dw'][c,0]=kernel*mm/131072
    net.weight['rgb_head'].zero_();net.weight['rgb_head'][:,:3,0,0]=w.new_tensor([[8,4.5,-3],[8,0,3],[8,-4.5,-3]])
    net.rgb_head_m.fill_(32768)


@torch.inference_mode()
def constant_check(net):
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9))
    z=torch.zeros(len(levels),16,1,1)
    for c,(a,b) in enumerate(GROUPS):
        z[:,a:b,0,0]=(levels[:,c,None]-2*torch.arange(b-a)[None]).clamp(0,2)
    out=fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]
    matrix=z.new_tensor([[16,9,-6],[16,0,6],[16,-9,-6]])
    # Round uses hardware's half-away-from-zero rule, not torch's ties-even.
    wanted=net.round(levels.float()@matrix.T+net.bias['rgb_head']).clamp(-128,127)
    err=float((out-wanted).abs().max());assert err<=1
    return dict(tested_group_levels=len(levels),unique_rgb=int(torch.unique(out,dim=0).shape[0]),max_error=err)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);initial=copy.deepcopy(net.state_dict())
    rows=m.load_data(args.data)['validation'];images={};records=[]
    for gain in (4,8,16):
        net.load_state_dict(initial);install(net,gain);checks=constant_check(net)
        sub=dest/f'gain{gain}';sub.mkdir();raw=net.export();(sub/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');tests=[]
        for row in rows:
            a=m.t.r.read(row['path']);x=torch.from_numpy(a.transpose(2,0,1).copy())[None].float()/128-1
            ours=net(x,all_nodes=True);ref,_=actual(x*128,True)
            err=max(float((v-ref[k]).abs().max()) for k,v in ours.items());assert err==0
            assert ref['up2_low_dw'].min()>=0 and ref['up2_low_dw'].max()<=2
            p=ref['rgb_head']/128;t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            loss,terms=joint_region_loss(p,t,labels);im=m.t.r.c.rgb_image(p[0]);im.save(sub/(row['id']+'.png'));images[gain,row['id']]=im
            tests.append(dict(id=row['id'],all_node_error=err,score=float(loss),**{k:float(v) for k,v in terms.items()}))
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng','bounded_training_model')}
        saved.update(model=copy.deepcopy(net.state_dict()),ternary_region_decoder=True,ternary_gain=gain,
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),added_training_steps=0,
            added_inference_operators=0,external_filter=False,visual_acceptance=False,
            independent_binary_decoder=False,group_native_count_decoder=None)
        torch.save(saved,sub/'initial.pt')
        records.append(dict(gain=gain,constant=checks,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=tests))
        print('TERNARY_REGION_SEED',gain,checks,{k:sum(r[k] for r in tests)/len(tests) for k in ('score','region_flatness','sharp_boundary')},flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');d=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[('target',Image.open(by_id[key]['target']))]+[(f'gain{g}',images[g,key]) for g in (4,8,16)]
        for j,(name,im) in enumerate(pics):
            board.paste(im.crop(box),(j*420,i*314+34));d.text((j*420+4,i*314+5),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(parent=args.checkpoint,records=records,
        added_training_steps=0,scope='Three-state field initialization; not learned visual acceptance'))


if __name__=='__main__':main()
