"""Six bounded multilevel brightness bands and centred chroma, fixed B1.

Correct-target injection is a required renderer diagnostic before training.
"""
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .audit_b1_region_renderer import tail
from .b1_package_oracle import oracle_records
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible

BASES=(0,16,32,52,76,100)
INCREMENTS=(16,16,20,24,24,28)


def encode_target(target):
    z=(target+1)*64;r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
    y=(r+2*g+b)/4;co=(r-b)/2+64;cg=(-r+2*g-b)/4+64
    return torch.cat([((y-lo)*127/inc).clamp(0,127) for lo,inc in zip(BASES,INCREMENTS)]+[co,cg],1)


@torch.no_grad()
def build():
    parent=m.OUT/'deep-regions-initial/initial.pt';state=torch.load(parent,map_location='cpu',weights_only=True)
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding']
    w,b=net.weight['res2_conv2__pw'],net.bias['res2_conv2__pw']
    raww=torch.stack([w[j].clone()/8 for j in (0,8,12)])
    rawb=torch.stack([b[j].clone()/8+v-8 for j,v in ((0,8),(8,49),(12,49))])
    w.zero_();b.zero_()
    for c,(lo,inc) in enumerate(zip(BASES,INCREMENTS)):w[c]=raww[0]*127/inc;b[c]=(rawb[0]-lo)*127/inc
    w[6:8]=raww[1:];b[6:8]=rawb[1:]
    began=False
    for key in net.weight:
        if key=='res3_conv1__dw':began=True
        if began:net.weight[key].zero_();net.bias[key].zero_()
    kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    # Clip signed residual outputs before any spatial averaging. Otherwise
    # negative outside-region logits incorrectly erode the positive region.
    for c in range(8):net.weight['res3_conv1__dw'][c,0,1,1]=1
    for c in range(8):
        net.weight['res3_conv1__pw'][c,c,0,0]=4/127 if c<6 else 1/3
        net.bias['res3_conv1__pw'][c]=0 if c<6 else 2/3
        net.weight['res3_conv2__dw'][c,0]=kernel
        net.weight['res3_conv2__pw'][c,c,0,0]=127/4 if c<6 else 3
        net.bias['res3_conv2__pw'][c]=0 if c<6 else -2
        net.weight['up1_low_dw'][c,0,1,1]=1
        net.weight['up1_low_pw'][c,c,0,0]=1
        net.weight['up1_high_dw'][c,0]=kernel
        net.weight['up2_low_dw'][c,0,1,1]=4/127 if c<6 else 1/3
        net.bias['up2_low_dw'][c]=0 if c<6 else 2/3
    w,b=net.weight['up2_low_pw'],net.bias['up2_low_pw']
    w[:3,:6,0,0]=w.new_tensor(INCREMENTS)[None]/4
    w[0,6,0,0]=3;w[0,7,0,0]=-3
    w[1,7,0,0]=3;b[1]=-66
    w[2,6:8,0,0]=-3;b[2]=132
    for c in range(3):net.weight['up2_high_dw'][c,0]=kernel
    sobel=w.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/4
    for c,k in zip(range(3,7),(sobel,-sobel,sobel.T,-sobel.T)):
        w[c,:6,0,0]=w.new_tensor(INCREMENTS)/4
        net.weight['up2_high_dw'][c,0]=k;net.bias['up2_high_dw'][c]=-1.5
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2
    net.weight['rgb_head'][:,3:7,0,0]=-.3;net.bias['rgb_head'].fill_(-128)
    for key,w in net.weight.items():
        if key=='res2_conv2__pw' or key.startswith('res3') or key.startswith('up') or key=='rgb_head':
            peak=w.abs().flatten(1).amax(1).clamp_min(1e-5);scale=torch.pow(2,torch.ceil(torch.log2(peak/127)))
            getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net,parent


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100697);dest=m.OUT/'multilevel-regions-initial';dest.mkdir(exist_ok=False)
    net,parent=build();raw=net.export();(dest/'initial.bin').write_bytes(raw)
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw));d=m.load_data();errors=[];pictures={};records=[]
    for row in d['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        a=net(x,all_nodes=True);b,_=actual(x*128,True);errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
        z=F.avg_pool2d(encode_target(target),4).round();z=torch.cat((z,z.new_zeros(z.shape[0],24,*z.shape[-2:])),1)
        oracle=tail(actual,z,'res2_add')/128;direct=tail(actual,z,'res3_add')/128
        pictures[row['id']]=[m.t.r.c.rgb_image(p[0]) for p in (target,a['rgb_head']/128,oracle,direct)]
        records.append(dict(id=row['id'],actual=m.t.r.metrics(a['rgb_head']/128,target),oracle_cleanup=m.t.r.metrics(oracle,target),oracle_decode=m.t.r.metrics(direct,target)))
    assert max(errors)==0
    params,residual=m.t.r.decode(raw);op,rp=oracle_records(params,residual)
    sample=np.random.default_rng(97).integers(-128,128,(12,12,3),dtype=np.int16);ref=m.t.r.execute_integer(m.t.r.contract_guard()[0],sample.tolist(),op,rp)
    got,_=actual(torch.tensor(sample.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32),True)
    error=max(float((v[0].permute(1,2,0).cpu()-torch.tensor(ref['tensors'][k])).abs().max()) for k,v in got.items());assert error==0
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=residual,contract_sha256=m.t.r.CONTRACT_SHA,parent=str(parent)),dest/'initial.pt')
    m.t.preview(net,d['validation'],dest/'preview');shutil.copy2(__file__,dest/'seed_source.py')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(pictures[key],['目标','实际初始化网络','正确区域：含清理','正确区域：只解码'])):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'renderer-diagnostic.png')
    write_json(dest/'review.json',dict(all_node_error=max(errors),oracle_error=error,records=records,sgd_steps=0,
        correct_region_panels_diagnostic_only=True,minimum_area_guarantee=False,code_sha256=file_hash(__file__)))
    print('MULTILEVEL_RENDERER',[(r['id'],round(r['oracle_cleanup']['mae'],3),round(r['oracle_decode']['mae'],3)) for r in records],flush=True)


if __name__=='__main__':main()
