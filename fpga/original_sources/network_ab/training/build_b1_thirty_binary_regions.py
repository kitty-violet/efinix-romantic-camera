"""Thirty binary region planes, joint colour before spatial reconstruction.

Use existing32 channels at180p for14/8/8 cumulative bits. Unlike one ternary
field, each changing bit has the same binary decision boundary. Existing DWs
open isolated positive islands, then one PW combines colours *before* the
remaining shared spatial kernels. No extra operators or storage are added.
Target injection below is a representation diagnostic, never source inference.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_ternary_field_targets import palette as ternary_palette
from .train_b1_ternary_regions import region_codes
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash

GROUPS30=((0,14),(14,22),(22,30))


def code_bits(codes):
    values=(codes//81,codes//9%9,codes%9)
    return torch.cat([(value[:,None]>torch.arange(b-a,device=codes.device)[None,:,None,None]).float()
        for value,(a,b) in zip(values,GROUPS30)],1)


@torch.no_grad()
def identity(net,key):
    net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    w=net.weight[key]
    if w.shape[-1]==3:w[:,0,1,1]=131071/131072
    else:
        for i in range(min(w.shape[:2])):w[i,i,0,0]=131071/131072


@torch.no_grad()
def install(net,opening=True):
    # res2.pw will be fitted as a30-plane classifier; discard its old skip
    # using the already-supported exact integer residual scale setting.
    for name in ('res2_add','res3_add'):
        net.package['residual_encoding'][name]=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    key='res3_conv1__dw';kernel=net.weight[key].new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    net.weight[key].zero_();net.weight[key][:30,0]=kernel/16;net.bias[key].zero_();getattr(net,key+'_m').fill_(8192)
    key='res3_conv1__pw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    for i in range(30):net.weight[key][i,i,0,0]=127*131071/131072;net.bias[key][i]=-63*127*131071/131072
    identity(net,'res3_conv2__dw');identity(net,'res3_conv2__pw');identity(net,'up1_low_dw')
    if opening:
        # For exactly0/127 bits these are exact3x3 erosion and dilation.
        net.weight['res3_conv2__dw'][:30,0].fill_(131071/131072)
        net.bias['res3_conv2__dw'][:30]=-8*127*131071/131072
        net.weight['up1_low_dw'][:30,0].fill_(131071/131072)
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_()
    for c,((a,b),increment) in enumerate(zip(GROUPS30,(8,12,12))):
        mult=round(131072*increment/127);getattr(net,key+'_m')[:,c]=mult
        net.weight[key][c,a:b,0,0]=mult/131072
    key='up1_high_dw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(8192)
    net.weight[key][:3,0]=kernel/16
    identity(net,'up2_low_dw');identity(net,'up2_low_pw')
    key='up2_high_dw';kernel=kernel.clone();kernel[1,1]=8;mult=round(131072/20)
    net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(mult);net.weight[key][:3,0]=kernel*mult/131072
    net.weight['rgb_head'].zero_();net.rgb_head_m.fill_(32768)
    net.weight['rgb_head'][:,:3,0,0]=kernel.new_tensor([[2,.75,-.5],[2,0,.5],[2,-.75,-.5]])


@torch.inference_mode()
def morphology_regression(net):
    z=torch.zeros(1,32,16,16);z[0,0,3,3]=127;z[0,1,8:13,8:13]=127
    out=fragment(net,z,'res3_conv1.pw')
    # Exact logical opening: the isolated point disappears, the5x5 survives.
    expected=-F.max_pool2d(-F.pad(z,(1,1,1,1),mode='replicate'),3,1)
    expected=F.max_pool2d(F.pad(expected,(1,1,1,1),mode='replicate'),3,1)
    assert torch.equal(out['up1_low_dw'],expected)
    assert out['up1_low_dw'][0,0].sum()==0 and int((out['up1_low_dw'][0,1]>0).sum())==25
    # The actual affine threshold is binary at every legal score.
    q=torch.arange(128).float();hard=torch.round((q-63)*127*131071/131072).clamp(0,127)
    assert torch.equal(hard,(q>=64).float()*127)
    return dict(logical_opening=True,threshold_codes=128,global_colour_min_area_guarantee=False)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);pk=ternary_palette(net)
    original=copy.deepcopy(net.state_dict());rows=m.load_data(args.data)['validation'];pictures={};records=[]
    for opening in (False,True):
        net.load_state_dict(original);install(net,opening);name='open3' if opening else 'plain'
        z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
        pal=fragment(net,z,'res3_conv1.pw')['rgb_head'][:,:,0,0]+128
        err=float((pal-torch.tensor(pk)).abs().max());assert err==0
        checks=morphology_regression(net) if opening else None
        sub=dest/name;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw)
        for row in rows:
            t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            codes=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
            bits=(F.avg_pool2d(code_bits(codes),4)>=.5).float()*127
            p=fragment(net,F.pad(bits,(0,0,0,0,0,2)),'res3_conv1.pw')['rgb_head']/128
            loss,values=joint_region_loss(p,t,labels);im=m.t.r.c.rgb_image(p[0]);pictures[name,row['id']]=im
            im.save(sub/(row['id']+'_injection.png'));records.append(dict(name=name,id=row['id'],loss=float(loss),**{k:float(v) for k,v in values.items()}))
        saved=dict(model=copy.deepcopy(net.state_dict()),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            contract_sha256=state['contract_sha256'],thirty_binary_regions=True,opening=opening,step=0,qat=True,
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),visual_acceptance=False,template_only=True)
        torch.save(saved,sub/'template.pt')
        print('THIRTY_BINARY_TEMPLATE',name,err,checks,flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');d=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,im) in enumerate([('target',Image.open(by_id[key]['target'])),('plain injection',pictures['plain',key]),('opening injection',pictures['open3',key])]):
            board.paste(im.crop(box),(420*j,i*314+34));d.text((420*j+4,i*314+5),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,source_prediction=False,
        target_injection=True,extra_inference_operations=0,constant_colour_error=0,visual_acceptance=False))


if __name__=='__main__':main()
