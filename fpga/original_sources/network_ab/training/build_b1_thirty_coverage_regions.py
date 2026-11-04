"""Preserve boundary phase in30 shared-colour region coverage planes.

Hard180p bit decisions visibly stair-step even with ideal labels. Instead keep
the4x4 native footprint coverage in the existing32-plane activation, combine
colour jointly, and reconstruct with common spatial kernels. This diagnostic
tests ideal coverage; source prediction still requires a fitted encoder.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .build_b1_thirty_binary_regions import install as binary_install,code_bits
from .probe_b1_ternary_field_targets import palette as ternary_palette
from .train_b1_ternary_regions import region_codes
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net):
    weights=copy.deepcopy(net.state_dict());residual=copy.deepcopy(net.package['residual_encoding'])
    binary_install(net,False)
    # Retain all three learned residual blocks, except res3.pw will become the
    #30-plane predictor. The old res2 residual arithmetic remains unchanged.
    current=net.state_dict()
    for key,value in weights.items():
        if key.startswith(('weight.res2_','weight.res3_','bias.res2_','bias.res3_','res2_','res3_')):
            current[key]=value
    net.load_state_dict(current);net.package['residual_encoding']['res2_add']=residual['res2_add']


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);pk=ternary_palette(net);install(net)
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    pal=fragment(net,z,'res3_add')['rgb_head'][:,:,0,0]+128
    assert torch.equal(pal,torch.tensor(pk))
    rows=m.load_data(args.data)['validation'];pictures={};records=[]
    for row in rows:
        t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        ids=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        coverage=F.avg_pool2d(code_bits(ids),4)
        for name,features in [('hard',(coverage>=.5).float()*127),('coverage',net.round(coverage*127))]:
            p=fragment(net,F.pad(features,(0,0,0,0,0,2)),'res3_add')['rgb_head']/128
            loss,values=joint_region_loss(p,t,labels);im=m.t.r.c.rgb_image(p[0]);pictures[name,row['id']]=im
            im.save(dest/f'{row["id"]}_{name}_injection.png')
            records.append(dict(name=name,id=row['id'],loss=float(loss),**{k:float(v) for k,v in values.items()}))
        print('THIRTY_COVERAGE_TARGET',row['id'],flush=True)
    saved=dict(model=copy.deepcopy(net.state_dict()),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
        contract_sha256=state['contract_sha256'],thirty_coverage_regions=True,step=0,qat=True,
        parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),visual_acceptance=False,template_only=True)
    torch.save(saved,dest/'template.pt');(dest/'coefficients-padded.bin').write_bytes(net.export())
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');d=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,im) in enumerate([('target',Image.open(by_id[key]['target'])),('hard180 injection',pictures['hard',key]),('coverage injection',pictures['coverage',key])]):
            board.paste(im.crop(box),(420*j,i*314+34));d.text((420*j+4,i*314+5),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,source_prediction=False,
        target_injection=True,extra_inference_operations=0,constant_colour_error=0,visual_acceptance=False))


if __name__=='__main__':main()
