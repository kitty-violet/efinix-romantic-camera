"""Compile a categorical colour decoder into unchanged B1 convolutions.

Known target masks are used ONLY to test decoder capacity. They are not
source inference, not a trained model, and not an accepted result.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .runtime import configure_workspace, write_json
from .engine import set_reproducible


@torch.no_grad()
def calibrate(net,keys):
    for key in keys:
        w=net.weight[key]
        scale=(w.abs().flatten(1).amax(1).clamp_min(1e-5)/120)
        getattr(net,key+'_m').copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def compile_decoder(net,weights,levels=4,aa=.25,preblur=0.):
    assert levels==4
    keys=['res3_conv2__dw','res3_conv2__pw','up1_low_dw','up1_low_pw','up1_high_dw',
          'up2_low_dw','up2_low_pw','up2_high_dw','rgb_head']
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for k in ('res3_conv2__dw','up1_low_dw','up1_high_dw','up2_low_dw'):
        net.weight[k][:,0,1,1]=1
    net.weight['up1_low_dw'][:,0]=kernel*preblur
    net.weight['up1_low_dw'][:,0,1,1]+=1-preblur
    for j in range(32):net.weight['res3_conv2__pw'][j,j,0,0]=1
    # Four exclusive levels per direction are packed in two stages, exactly
    # matching the existing32->16->8 feature widths. No LUT/argmax at runtime.
    for j in range(8):
        net.weight['up1_low_pw'][2*j,4*j,0,0]=.5
        net.weight['up1_low_pw'][2*j,4*j+1,0,0]=1
        net.weight['up1_low_pw'][2*j+1,4*j+2,0,0]=.75
        net.weight['up1_low_pw'][2*j+1,4*j+3,0,0]=1
        net.weight['up2_low_pw'][j,2*j,0,0]=.5
        net.weight['up2_low_pw'][j,2*j+1,0,0]=1
    net.weight['up2_high_dw'][:,0]=kernel*aa*16/127
    net.weight['up2_high_dw'][:,0,1,1]+=(1-aa)*16/127
    net.weight['rgb_head'][:,:,0,0]=weights[:8].T/16
    net.bias['rgb_head'][:]=weights[8]-128
    calibrate(net,keys)


@torch.no_grad()
def native_categorical(target,palette):
    flat=target.permute(0,2,3,1).reshape(-1,3);result=[]
    for start in range(0,len(flat),8192):
        x=flat[start:start+8192]
        result.append((x.square().sum(1,keepdim=True)+palette.square().sum(1)[None]-2*x@palette.T).argmin(1))
    ids=torch.cat(result).reshape(target.shape[0],*target.shape[2:])
    # Class0 is the shared base colour. All other class coverages are explicit.
    hot=F.one_hot(ids,len(palette))[...,1:].permute(0,3,1,2).float()
    return F.avg_pool2d(hot,4)


@torch.no_grad()
def actual_palette(net):
    z=torch.zeros(33,32,4,4,device='cuda')
    for j in range(32):z[j+1,j]=127
    return fragment(net,z)['rgb_head'][:,:,8,8]/128


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--palette',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006251);torch.set_num_threads(2)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding']
    spec=json.loads(Path(a.palette).read_text())['candidates']['4'];weights=torch.tensor(spec['weights'],device='cuda')
    sample=m.t.r.tensor(m.t.r.read(m.load_data()['validation'][0]['path']))[None,...,:64,:64]
    records=[];images={};rows={r['id']:r for r in m.load_data()['validation']}
    for name,aa,preblur in [('point',0.,0.),('light-aa',.5,0.),('shape-aa',.5,.5)]:
        compile_decoder(net,weights,aa=aa,preblur=preblur)
        full=net(sample,all_nodes=True);part=fragment(net,full['res3_conv1.pw'])
        errors={k:float((v-full[k]).abs().max()) for k,v in part.items() if k not in ('res2_add','res3_conv1.pw')}
        assert max(errors.values())==0
        raw=net.export();actual=IntegerB1(runtime_from_image(raw));got,_=actual(sample*128,True)
        error=max(float((v-got[k]).abs().max()) for k,v in full.items());assert error==0
        del full,part,got,actual
        pal=actual_palette(net)
        for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
            target=m.t.r.tensor(m.t.r.read(rows[key]['target'])[box[1]:box[3],box[0]:box[2]])[None]
            cov=native_categorical(target,pal);p=fragment(net,(cov*127).round())['rgb_head']/128
            im=m.t.r.c.rgb_image(p[0]);images[(i,name)]=im;im.save(dest/(str(i)+'_'+name+'_target_injection.png'))
            records.append(dict(crop=label,variant=name,mae=float((p-target).abs().mean()*128)))
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=state['residual_encoding'],
                        field_node='res2_add',fixed_sampling=state.get('fixed_sampling',False),
                        categorical_levels=4,coverage_layout='32exclusive_8rays',parent=a.warm,
                        palette_spec=spec,decoder_diagnostic_only=True),dest/(name+'.pt'))
        print('CATEGORICAL_DECODER',name,error,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[Image.open(rows[key]['target']).crop(box)]+[images[(i,n)] for n in ('point','light-aa','shape-aa')]
        for j,(im,title) in enumerate(zip(ims,('离线目标','点采样·仅注入诊断','轻边界AA·仅注入诊断','形状AA·仅注入诊断'))):
            board.paste(im,(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'results.json',dict(records=records,small_node_error=error,target_injection=True,source_inference=False,
        trained=False,visual_acceptance=False,added_operators=0,bytes=len(raw)))


if __name__=='__main__':main()
