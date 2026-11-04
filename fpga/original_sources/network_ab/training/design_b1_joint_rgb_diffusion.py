"""Three joint-RGB edge-stopping iterations compiled into unchanged B1.

Unlike independent colour/luma filters, each neighbour uses the same RGB
conductance (proved below for legal integer differences at threshold8). The
decoder uses identical positive spatial kernels on every colour field. This
is an explicit coefficient design, not a learned/accepted FPGA result.
"""
import argparse,copy,itertools,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .design_b1_edge_stopping_diffusion import clear,identity_dw,install as luma_install,KERNEL
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m

DIRS=((-1,0),(1,0),(0,-1),(0,1))


def slot(direction,colour,sign):return direction*6+colour*2+sign


@torch.no_grad()
def install(net,threshold=8,spatial_mix=.25):
    luma_install(net,gain=16,decode_gain=1)
    clear(net,'down2__pw')
    for direction in range(4):
        for colour in range(3):
            for sign in range(2):net.weight['down2__pw'][slot(direction,colour,sign),colour,0,0]=1
    for colour in range(3):net.weight['down2__pw'][24+colour,colour,0,0]=1
    for r in range(1,4):
        dw=f'res{r}_conv1__dw';pw=f'res{r}_conv1__pw';clear(net,dw);clear(net,pw,1/8)
        for direction,(dy,dx) in enumerate(DIRS):
            for colour in range(3):
                for sign in range(2):
                    c=slot(direction,colour,sign);sgn=1 if sign==0 else -1
                    net.weight[dw][c,0,1+dy,1+dx]=sgn;net.weight[dw][c,0,1,1]=-sgn
                    # relu(+/-d_c + 8*(threshold - sum(abs(d_RGB)))).
                    for other_colour in range(3):
                        for other_sign in range(2):net.weight[pw][c,slot(direction,other_colour,other_sign),0,0]=-8
                    net.weight[pw][c,slot(direction,colour,0),0,0]+=sgn
                    net.weight[pw][c,slot(direction,colour,1),0,0]-=sgn
                    net.bias[pw][c]=8*threshold
        for colour in range(3):
            net.weight[dw][24+colour,0,1,1]=1;net.weight[pw][24+colour,24+colour,0,0]=1
        identity_dw(net,f'res{r}_conv2__dw',range(27));pw=f'res{r}_conv2__pw';clear(net,pw)
        for out in range(27):
            colour=(out%6)//2 if out<24 else out-24
            net.weight[pw][out,24+colour,0,0]=1
            for direction in range(4):
                net.weight[pw][out,slot(direction,colour,0),0,0]=1/8
                net.weight[pw][out,slot(direction,colour,1),0,0]=-1/8
    identity_dw(net,'up1_low_dw',range(27));k='up1_low_pw';clear(net,k)
    for c in range(14):
        getattr(net,k+'_m')[:,c]=65536
        net.weight[k][c,24:27,0,0]=torch.tensor([8.,16,8]);net.bias[k][c]=63.5-32*(12+8*c)
    net.weight[k][14,24:27,0,0]=torch.tensor([.5,0,-.5]);net.bias[k][14]=64
    net.weight[k][15,24:27,0,0]=torch.tensor([-.25,.5,-.25]);net.bias[k][15]=64
    # Shared convex interpolation cannot introduce an independently shifted
    # luminance/chroma sharpening ramp. Only the same three existing DW taps.
    kernel=KERNEL*spatial_mix;kernel[1,1]+=1-spatial_mix
    identity_dw(net,'up1_high_dw',range(16),kernel)
    identity_dw(net,'up2_low_dw',range(16))
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert torch.equal(w,net.round(w/s)*s) and (w/s).abs().max()<=127,(k,'quantization')


def flux(d,threshold=8):
    gate=8*(threshold-d.abs().sum(1,keepdim=True))
    return ((d+gate).clamp(0,127)-(-d+gate).clamp(0,127))/2


def reference_step(rgb,threshold=8):
    pad=F.pad(rgb,(1,1,1,1),mode='replicate');h,w=rgb.shape[-2:];delta=torch.zeros_like(rgb)
    for dy,dx in DIRS:delta+=flux(pad[...,1+dy:1+dy+h,1+dx:1+dx+w]-rgb,threshold)/4
    return m.t.r.c.Round()(rgb+delta).clamp(0,127)


@torch.no_grad()
def regression(net,threshold):
    torch.manual_seed(81014);rgb=torch.randint(0,128,(2,3,28,32)).float();z=torch.zeros(2,32,28,32)
    for out in range(27):z[:,out]=rgb[:,(out%6)//2 if out<24 else out-24]
    actual=fragment(net,z,'down2.pw');ref=rgb
    for r in range(1,4):
        ref=reference_step(ref,threshold);assert torch.equal(actual[f'res{r}_add'][:,24:27],ref)
    for c in ([0,0,0],[24,83,109],[127,127,127]):
        const=torch.tensor(c).float()[None,:,None,None].expand_as(rgb)
        assert torch.equal(reference_step(const,threshold),const)
    pad=F.pad(rgb,(1,1,1,1),mode='replicate');lo=-F.max_pool2d(-pad,3,1);hi=F.max_pool2d(pad,3,1)
    out=reference_step(rgb,threshold);assert ((out>=lo)&(out<=hi)).all()
    # For T=8, all S>=10 have zero flux algebraically; exhaust the remaining
    # RGB difference triples, including signs. Common conductance means no
    # vector rotation before the explicitly measured integer rounding.
    ds=torch.tensor(list(itertools.product(range(-10,11),repeat=3))).float()[:,:,None,None]
    f=flux(ds,threshold);cross=torch.cross(ds[...,0,0],f[...,0,0],dim=1)
    cross_max=float(cross.abs().max())
    if threshold==8:assert cross_max==0
    print('JOINT_RGB_DIFFUSION_PASS real3iterations/constant/range; flux_cross_max',cross_max,flush=True)
    return dict(actual_three_iteration_error=0,constant_error=0,local_extrema_error=0,flux_cross_max=cross_max,
                common_rgb_conductance_exact_before_rounding=threshold==8)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--threshold',type=int,choices=(8,12),default=8)
    a=ap.parse_args();torch.set_num_threads(2);state,net=load(a.checkpoint);install(net,a.threshold)
    checks=regression(net,a.threshold);data=m.load_data(a.data);out=Path(a.out);out.mkdir(exist_ok=False)
    raw=net.export();assert len(raw)==20480;(out/'coefficients-padded.bin').write_bytes(raw)
    rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={};records=[]
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
        im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(out/(row['id']+'_output.png'));ims[row['id']]=im
        records.append(dict(id=row['id'],actual_all_nodes_error=0))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);rows={r['id']:r for r in data['validation']}
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(title,im) in enumerate((('原片',Image.open(rows[rid]['path'])),('固定目标',Image.open(rows[rid]['target'])),('共同RGB扩散 · 实际B1',ims[rid]))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    saved=dict(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
        contract_sha256=state['contract_sha256'],step=0,qat=False,template_only=True,designed_not_trained=True,
        visual_acceptance=False,designed_joint_rgb_diffusion=dict(threshold=a.threshold,iterations=3,spatial_mix=.25),
        training_provenance=dict(kind='designed coefficients, not gradient training',code_sha256=file_hash(__file__)))
    torch.save(saved,out/'designed.pt')
    write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),
        coefficient_sha256=file_hash(out/'coefficients-padded.bin'),designed_not_trained=True,visual_acceptance=False,
        extra_inference_operations=0,whole_decoder_hue_preservation_not_proven=True))
    print('JOINT_RGB_DESIGN_COMPLETE all6nodes0 20KiB',flush=True)


if __name__=='__main__':main()
