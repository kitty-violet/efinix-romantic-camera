"""Form three-state fields before the last local vote, using existing layers.

Moves the existing affine clipping into the180p PW. The first360p DW then
rounds region coverage to trits; the next DW votes on actual trits. Distinct
from the older RGB-coordinate quantizer-order probe. This changes the function
and must be visually tested; constant1215-colour palette must stay identical.
"""
import argparse,copy
from pathlib import Path
import torch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ternary_field_targets import canonical_states
from .train_b1_projected_count_regions import region_geometry
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net,mode):
    gain=net.weight['up1_high_dw'][:15,0].sum((-1,-2)).clone()
    offset=net.bias['up1_high_dw'][:15].clone()
    assert (gain>0).all() and (net.weight['up1_high_dw'][:15]>=0).all()
    net.weight['up1_low_pw'][:15]*=gain[:,None,None,None]
    net.bias['up1_low_pw'][:15]=net.bias['up1_low_pw'][:15]*gain+offset
    net.weight['up1_high_dw'][:15]*=(2/(127*gain))[:,None,None,None]
    net.bias['up1_high_dw'].zero_()
    kernel=net.weight['up2_low_dw'].new_ones(3,3)/9
    if mode=='identity':kernel.zero_();kernel[1,1]=1
    elif mode=='binomial':kernel=kernel.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['up2_low_dw'].zero_();net.bias['up2_low_dw'].zero_()
    net.weight['up2_low_dw'][:15,0]=kernel
    for key in ('up1_low_pw','up1_high_dw','up2_low_dw'):
        w=net.weight[key];mult=(w.abs().flatten(1).amax(1)*131072/126).ceil().clamp(1,131071)
        assert (w.abs().flatten(1).amax(1)<=127*mult/131072+1e-5).all()
        getattr(net,key+'_m').copy_(mult.double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    base=m.t.r.c.ColourB1(revive=False);base.load_state_dict(state['model']);base.quantized=True
    base.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    variants={'base':base};data=m.load_data(a.data);records=[];images={}
    codes=torch.arange(1215)[:,None,None];fields=torch.nn.functional.pad(canonical_states(codes),(0,0,0,0,0,1))
    target_palette=fragment(base,fields,'up2_low_dw')['rgb_head']
    for mode in ('identity','box','binomial'):
        net=copy.deepcopy(base);install(net,mode)
        latent=net.round(fields*127/2)
        colours=fragment(net,latent,'up1_low_pw')['rgb_head']
        assert torch.equal(colours[:,:,0,0],target_palette[:,:,0,0]),(mode,(colours[:,:,0,0]-target_palette[:,:,0,0]).abs().max())
        variants[mode]=net;sub=dest/mode;sub.mkdir()
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
        saved.update(model=net.state_dict(),ternary_early_vote=mode,visual_acceptance=False,training_steps_since_vote=0)
        torch.save(saved,sub/'initial.pt');(sub/'initial.bin').write_bytes(net.export())
    for name,net in variants.items():
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);other,_=actual(x*128,True)
            err=max(float((v-other[k]).abs().max()) for k,v in nodes.items());assert err==0
            z=nodes['up2_low_dw'][0,:15];assert z.min()>=0 and z.max()<=2
            code=(81*z[:7].sum(0)+9*z[7:11].sum(0)+z[11:15].sum(0)).long().numpy()
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);images[name,row['id']]=im;im.save(dest/f'{name}_{row["id"]}.png')
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()-128
            records.append(dict(variant=name,id=row['id'],all_node_error=err,mae=float((nodes['rgb_head']-target).abs().mean()),**region_geometry(code)))
        print('TERNARY_VOTE_ORDER',name,flush=True)
    byid={r['id']:r for r in data['validation']};board=Image.new('RGB',(2100,314*len(m.t.r.c.CROPS)),'#17202a')
    draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('target',Image.open(byid[key]['target']))]+[(name,images[name,key]) for name in variants]
        for j,(title,im) in enumerate(choices):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),
        constant1215_palette_equal=True,extra_inference_ops=0,bytes=20480,visual_acceptance=False))


if __name__=='__main__':main()
