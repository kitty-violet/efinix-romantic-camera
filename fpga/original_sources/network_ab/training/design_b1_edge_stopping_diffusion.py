"""Compile bounded edge-stopping luminance diffusion into existing B1 weights.

Designed coefficient experiment, not learned weights or a quality release.
Uses seven existing DW/PW pairs, no runtime loop/operator/filter. Two clipped
ramps realize a redescending scalar flux, sharing luminance geometry across RGB.
Inspired by Perona--Malik/TNRD; not either paper's method or performance claim.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m

DIRS=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
KERNEL=torch.tensor([[1.,2,1],[2,4,2],[1,2,1]])/16


@torch.no_grad()
def clear(net,key,quantum=1/64):
    net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(round(131072*quantum))


@torch.no_grad()
def identity_dw(net,key,channels,kernel=None):
    clear(net,key)
    for c in channels:
        if kernel is None:net.weight[key][c,0,1,1]=1
        else:net.weight[key][c,0]=kernel


@torch.no_grad()
def diffusion_dw(net,key,gain):
    clear(net,key,gain/64)
    for i,(dy,dx) in enumerate(DIRS):
        for b,g in enumerate((gain,gain/2)):
            c=2*i+b;net.weight[key][c,0,1+dy,1+dx]=g
            net.weight[key][c,0,1,1]-=g;net.bias[key][c]=64
    for c in (16,17,18):net.weight[key][c,0,1,1]=1


def y_terms(gain,dt):
    terms={16:1.}
    for i in range(8):terms[2*i]=2*dt/gain;terms[2*i+1]=-2*dt/gain
    return terms


@torch.no_grad()
def diffuse_pw(net,key,gain,dt):
    clear(net,key,1/64)
    for c in range(17):
        for i,v in y_terms(gain,dt).items():net.weight[key][c,i,0,0]=v
    net.weight[key][17,17,0,0]=1;net.weight[key][18,18,0,0]=1


@torch.no_grad()
def install(net,blocks=True,diffuse=True,decode_gain=2,gain=8):
    # Every coefficient is explicitly reconstructed. No inherited partial head.
    for k in net.weight:clear(net,k)
    k='stem'
    for c in range(3):net.weight[k][c,c,1,1]=.5;net.bias[k][c]=64
    identity_dw(net,'down1__dw',range(3),KERNEL)
    for c in range(3):net.weight['down1__pw'][c,c,0,0]=1
    identity_dw(net,'down2__dw',range(3),KERNEL)
    for c in range(17):net.weight['down2__pw'][c,:3,0,0]=torch.tensor([.25,.5,.25])
    net.weight['down2__pw'][17,:3,0,0]=torch.tensor([.5,0,-.5]);net.bias['down2__pw'][17]=64
    net.weight['down2__pw'][18,:3,0,0]=torch.tensor([-.25,.5,-.25]);net.bias['down2__pw'][18]=64
    dt=1/8 if diffuse else 0.
    for r in range(1,4):
        for c in (1,2):
            diffusion_dw(net,f'res{r}_conv{c}__dw',gain)
            diffuse_pw(net,f'res{r}_conv{c}__pw',gain,dt)
        # Existing legal residual encoding suppresses the old skip to zero.
        net.package['residual_encoding'][f'res{r}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    diffusion_dw(net,'up1_low_dw',gain)
    k='up1_low_pw';clear(net,k)
    if blocks:
        for c in range(14):
            getattr(net,k+'_m')[:,c]=65536
            for i,v in y_terms(gain,dt).items():net.weight[k][c,i,0,0]=32*v
            net.bias[k][c]=63.5-32*(12+8*c)
        net.weight[k][14,17,0,0]=1;net.weight[k][15,18,0,0]=1
        identity_dw(net,'up1_high_dw',range(16),KERNEL)
        identity_dw(net,'up2_low_dw',range(16))
        # Existing activation compresses the coverage ramp jointly for every Y
        # level. This is a tested variant, not an assertion it removes jaggies.
        net.weight['up2_low_dw'][:14,0,1,1]=decode_gain
        net.bias['up2_low_dw'][:14]=63.5*(1-decode_gain)
        if decode_gain>1:getattr(net,'up2_low_dw_m')[:,:14]=16384
        k='up2_low_pw';clear(net,k,1/16)
        net.weight[k][:3,:14,0,0]=1/16
        net.weight[k][0,14,0,0]=1;net.weight[k][0,15,0,0]=-1;net.bias[k][0]=8
        net.weight[k][1,15,0,0]=1;net.bias[k][1]=-56
        net.weight[k][2,14,0,0]=-1;net.weight[k][2,15,0,0]=-1;net.bias[k][2]=136
    else:
        for i,v in y_terms(gain,dt).items():net.weight[k][0,i,0,0]=v
        net.weight[k][1,17,0,0]=1;net.weight[k][2,18,0,0]=1
        identity_dw(net,'up1_high_dw',range(3),KERNEL);identity_dw(net,'up2_low_dw',range(3))
        k='up2_low_pw';clear(net,k,1/64)
        net.weight[k][:3,:3,0,0]=torch.tensor([[1.,1,-1],[1,0,1],[1,-1,-1]])
        net.bias[k][:3]=torch.tensor([0.,-64,128])
    identity_dw(net,'up2_high_dw',range(3),KERNEL)
    clear(net,'rgb_head',1/32)
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2
    net.bias['rgb_head'].fill_(-128);net.quantized=True
    # Validate every effective coefficient lies exactly on its deployed lattice.
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert torch.equal(w,net.round(w/s)*s),(k,'nonrepresentable')
        assert (w/s).abs().max()<=127,(k,'range')


def reference_update(y,gain=8,dt=1/8,rounded=True):
    pad=F.pad(y,(1,1,1,1),mode='replicate');h,w=y.shape[-2:];flux=torch.zeros_like(y)
    rnd=m.t.r.c.Round()
    for dy,dx in DIRS:
        d=pad[...,1+dy:1+dy+h,1+dx:1+dx+w]-y
        a=rnd(64+gain*d).clamp(0,127);b=rnd(64+gain/2*d).clamp(0,127)
        flux+=2/gain*(a-b)
    update=y+dt*flux
    return rnd(update).clamp(0,127) if rounded else update


@torch.no_grad()
def regression(net,gain=8):
    torch.manual_seed(2026100812)
    y=torch.randint(0,128,(2,1,28,32)).float();z=torch.zeros(2,32,28,32)
    z[:,:17]=y;z[:,17:19]=64
    part=fragment(net,z,'down2.pw');ref=y
    for r in range(1,4):
        for c in (1,2):
            ref=reference_update(ref,gain=gain)
            assert torch.equal(part[f'res{r}_conv{c}.pw'][:,16:17],ref)
        assert torch.equal(part[f'res{r}_add'][:,16:17],ref)
    # The final PW fuses diffusion and thresholds: there is no intermediate
    # luma requantization between them in the real graph.
    ref=reference_update(ref,gain=gain,rounded=False)
    thresholds=torch.arange(14)[None,:,None,None]*8+12
    expected=net.round(32*(ref-thresholds)+63.5).clamp(0,127)
    assert torch.equal(part['up1_low_pw'][:,:14],expected)
    # Constants, strong jumps, and no new extrema are actual pair-level checks.
    for level in (0,8,64,120,127):assert torch.equal(reference_update(torch.full_like(y,level),gain=gain),torch.full_like(y,level))
    step=torch.full_like(y,24);step[...,16:]=100;assert torch.equal(reference_update(step,gain=gain),step)
    p=F.pad(y,(1,1,1,1),mode='replicate');lo=-F.max_pool2d(-p,3,1);hi=F.max_pool2d(p,3,1)
    after=reference_update(y,gain=gain);assert ((after>=lo)&(after<=hi)).all()
    fine=torch.full_like(y,64);fine[...,::2,::2]=68;filtered=fine
    for _ in range(7):filtered=reference_update(filtered,gain=gain)
    assert filtered.std()<fine.std()
    print('EDGE_STOPPING_PASS actual7pairs/constant/strong_step/local_extrema/fine_variation',flush=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--gain',type=int,choices=(8,16),default=8);ap.add_argument('--only-blocks',action='store_true')
    a=ap.parse_args();torch.set_num_threads(2);state,base=load(a.checkpoint)
    check=copy.deepcopy(base);install(check,gain=a.gain);regression(check,a.gain)
    out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data);pictures={};records=[]
    variants=[('linear_blocks',True,False,2),('diffusion_continuous',False,True,1),('diffusion_blocks',True,True,2)]
    if a.only_blocks:variants=variants[-1:]
    for name,blocks,diffuse,dg in variants:
        net=copy.deepcopy(base);install(net,blocks,diffuse,dg,gain=a.gain);raw=net.export();assert len(raw)==20480
        sub=out/name;sub.mkdir();(sub/'coefficients-padded.bin').write_bytes(raw)
        integer=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=integer(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'))
            pictures[name,row['id']]=im;records.append(dict(variant=name,id=row['id'],actual_all_nodes_error=0))
        saved={k:copy.deepcopy(v) for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
        saved.update(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            step=0,designed_not_trained=True,template_only=True,qat=False,visual_acceptance=False,
            designed_diffusion=dict(gain=a.gain,dt=1/8 if diffuse else 0,blocks=blocks,decode_gain=dg),
            training_provenance=dict(kind='designed edge-stopping diffusion coefficients, no gradient training',code_sha256=file_hash(__file__)))
        torch.save(saved,sub/'designed.pt');print('EDGE_STOPPING_VARIANT',name,'6 actual nodes0',flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);rows={r['id']:r for r in data['validation']}
    for name,_,_,_ in variants:
        board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
        for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
            for j,(title,im) in enumerate((('原片',Image.open(rows[rid]['path'])),('固定目标',Image.open(rows[rid]['target'])),(name+' 实际B1',pictures[name,rid]))):
                board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
        board.save(out/name/'comparison.png')
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),records=records,source_sha256=file_hash(__file__),
        designed_not_trained=True,extra_inference_operations=0,visual_acceptance=False,
        reference='https://authors.library.caltech.edu/records/1p8h5-5x870',not_original_paper_reproduction=True))


if __name__=='__main__':main()
