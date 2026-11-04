"""Delay ternary group-count decisions from360 to720 using existing8 rows.

Controlled reuse of earlier binary pair packing, now preserving15/9/9 levels
and the current fitted encoder. Two continuous fields share one existing row;
its0..4 decision occurs after the final nearest upsample. No new hardware op.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ternary_field_targets import palette,canonical_states
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import joint_region_loss
from .audit_b1_white_rims_grid import measures
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash

PACKS=((0,1),(2,3),(4,5),(6,),(7,8),(9,10),(11,12),(13,14))


@torch.no_grad()
def install(net):
    pal=torch.tensor(palette(net));origin=7*81+4*9+4
    centre=pal[origin];matrix=torch.stack((pal[origin+81]-centre,pal[origin+9]-centre,pal[origin+1]-centre),1)
    bias=centre-128-matrix@matrix.new_tensor([7,4,4])
    codes=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    assert torch.equal((codes@matrix.T+bias+128).clamp(0,255),pal)
    # The existing360 spatial vote keeps fractional coverage in0..120.
    key='up2_low_dw';w=net.weight[key];b=net.bias[key];w.mul_(60);b.mul_(60)
    mult=(w.abs().flatten(1).amax(1)*131072/126).ceil().clamp(1,131071);getattr(net,key+'_m').copy_(mult.double()[None,:,None,None])
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):net.weight[key].zero_();net.bias[key].zero_()
    net.up2_low_pw_m.fill_(65536)
    k=matrix.new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    for c,indices in enumerate(PACKS):
        net.weight['up2_low_pw'][c,list(indices),0,0]=1/len(indices)
        multiplier=round(131072*len(indices)/(16*60));net.up2_high_dw_m[:,c]=multiplier
        net.weight['up2_high_dw'][c,0]=k*multiplier/131072
        group=0 if c<4 else 1 if c<6 else 2
        net.weight['rgb_head'][:,c,0,0]=matrix[:,group]
    net.rgb_head_m.fill_(65536);net.bias['rgb_head'].copy_(bias)
    return dict(matrix=matrix.tolist(),bias=bias.tolist(),levels=[5,5,5,3,5,5,5,5])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);pk=palette(base);net=copy.deepcopy(base);setup=install(net)
    code=torch.arange(1215)[:,None,None];fields=F.pad(canonical_states(code)*63.5,(0,0,0,0,0,1)).round()
    got=fragment(net,fields,'up1_high_dw')['rgb_head'][:,:,0,0]+128
    assert torch.equal(got,torch.tensor(pk)),float((got-torch.tensor(pk)).abs().max())
    # Include arbitrary within-group field permutations, not only canonical.
    g=torch.Generator().manual_seed(8);random=torch.randint(0,3,(4096,16,1,1),generator=g).float();random[:,15]=0
    desired=(random[:,:7].sum(1)*81+random[:,7:11].sum(1)*9+random[:,11:15].sum(1)).long()[:,0,0]
    got=fragment(net,(random*63.5).round(),'up1_high_dw')['rgb_head'][:,:,0,0]+128
    assert torch.equal(got,torch.tensor(pk)[desired])
    raw=net.export();assert len(raw)==20480;(dest/'initial.bin').write_bytes(raw)
    saved=copy.deepcopy(state);saved.update(model=net.state_dict(),ternary_packed720=True,packed720=setup,visual_acceptance=False,added_inference_ops=0,step=0)
    torch.save(saved,dest/'initial.pt');actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');rows=m.load_data(a.data)['validation'];images={};records=[]
    for row in rows:
        def read(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()
        src,target=read(row['path']),read(row['target']);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        nodes=net(src/128-1,all_nodes=True);other,_=actual(src-128,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        ids=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None];canonical=canonical_states(ids)
        # Invert the unchanged channel-specific gauge of the high360 clip.
        ww=net.weight['up1_high_dw'][:15].sum((1,2,3))[None,:,None,None];bb=net.bias['up1_high_dw'][:15][None,:,None,None]
        latent=F.avg_pool2d((63.5+128*(canonical-1)-bb)/ww,4)
        latent=F.pad(net.round(latent).clamp(0,127),(0,0,0,0,0,1));injected=fragment(net,latent,'up1_low_pw')['rgb_head']/128
        variants={'base':base(src/128-1)[0],'packed720':nodes['rgb_head']/128,'target injection':injected}
        record=dict(id=row['id'],variants={})
        for name,p in variants.items():
            _,v=joint_region_loss(p,target/128-1,labels);stats,_=measures(p*128+128,target,src);record['variants'][name]=dict(**{k:float(z) for k,z in v.items()},**stats)
            im=m.t.r.c.rgb_image(p[0]);images[row['id'],name]=im;im.save(dest/(row['id']+'_'+name.replace(' ','_')+'.png'))
        records.append(record);print('TERNARY_PACKED720',row['id'],{k:(v['bright_target_excess_fraction'],v['sharp_boundary']) for k,v in record['variants'].items()},flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);byid={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('target',Image.open(byid[key]['target']))]+[(name,images[key,name]) for name in ('base','packed720','target injection')]
        for j,(name,im) in enumerate(choices):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,constant_palette_error=0,random_field_permutations=4096,all_node_error=0,
        added_inference_ops=0,byte_size=20480,trained_after_change=False,visual_acceptance=False,source_checkpoint_sha256=file_hash(a.checkpoint)))


if __name__=='__main__':main()
