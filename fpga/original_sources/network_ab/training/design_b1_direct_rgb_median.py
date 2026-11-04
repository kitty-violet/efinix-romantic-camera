"""Direct RGB median and 216-colour ordered decisions in unchanged B1.

Controlled alternative to the old YCoCg median/radix designs: every RGB
output is a positive sum of its own ordered predicates, so independently
shifted luma/chroma boundaries cannot manufacture a rainbow by cancellation.
This does not prove hue preservation or sufficient natural-picture quality.
All processing is existing DW/PW/clip arithmetic in the coefficient package.
"""
import argparse,copy,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .build_b1_continuous_median import build as median_build
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,gain=16):
    median_build(net)
    clear(net,'stem')
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    # Exact 2x2 area footprint, no repeated 3x3 blur before decimation.
    box=torch.zeros(3,3);box[1:,1:]=.25
    identity_dw(net,'down1__dw',range(3),box);identity_dw(net,'down2__dw',range(3),box)
    k='up1_low_pw';clear(net,k)
    levels=np.array([4,28,52,76,100,124],float)
    thresholds=(levels[:-1]+levels[1:])/2
    for c in range(3):
        for j,threshold in enumerate(thresholds):
            net.weight[k][c*5+j,c,0,0]=1;net.bias[k][c*5+j]=63.5-threshold
    identity_dw(net,'up1_high_dw',range(15),KERNEL)
    clear(net,'up2_low_dw',.25)
    for c in range(15):
        net.weight['up2_low_dw'][c,0,1,1]=gain;net.bias['up2_low_dw'][c]=63.5*(1-gain)
    k='up2_low_pw';clear(net,k,1/512)
    # Exact representable increment 24/128; reachable maxima are 123 rather
    # than124 after five 127-sized fields. Record, do not hide this rounding.
    for c in range(3):net.weight[k][c,c*5:c*5+5,0,0]=24/128;net.bias[k][c]=4
    identity_dw(net,'up2_high_dw',range(3),KERNEL)
    clear(net,'rgb_head',1/32)
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2
    net.bias['rgb_head'][:]=torch.tensor([-131.,-128.,-125.])
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert (w/s).abs().max()<=127,(k,'range')
    net.quantized=True


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);state,net=load(a.checkpoint);install(net)
    # The existing median network is tested on independent random RGB fields.
    torch.manual_seed(8182);rgb=torch.randint(0,128,(3,3,32,40)).float();z=torch.zeros(3,32,32,40)
    for c in range(3):z[:,4*c:4*c+3]=rgb[:,c:c+1]
    part=fragment(net,z,'down2.pw')
    f=torch.nn.functional
    expected=f.pad(rgb,(1,1,0,0),mode='replicate').unfold(-1,3,1).median(-1).values
    expected=f.pad(expected,(0,0,1,1),mode='replicate').unfold(-2,3,1).median(-1).values
    assert torch.equal(expected,part['res3_add'][:,:3])
    raw=net.export();assert len(raw)==20480;(out/'coefficients-padded.bin').write_bytes(raw)
    rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');records=[];pictures={}
    data=m.load_data(a.data)
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
        im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(out/(row['id']+'_output.png'));pictures[row['id']]=im
        records.append(dict(id=row['id'],actual_all_nodes_error=0))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    rows={r['id']:r for r in data['validation']}
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(title,im) in enumerate((('原片',Image.open(rows[rid]['path'])),('原目标',Image.open(rows[rid]['target'])),('设计权重·实际整数B1',pictures[rid]))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png')
    saved=dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],
        designed_not_trained=True,qat=False,step=0,visual_acceptance=False,extra_inference_operators=0)
    torch.save(saved,out/'designed.pt')
    for path in (Path(__file__),Path(__file__).with_name('build_b1_continuous_median.py')):shutil.copy2(path,out/path.name)
    write_json(out/'review.json',dict(args=vars(a),records=records,median_actual_error=0,byte_size=20480,
        coefficient_sha256=file_hash(out/'coefficients-padded.bin'),direct_rgb_positive_reconstruction=True,
        visual_acceptance=False,designed_not_trained=True))
    print('DIRECT_RGB_MEDIAN_COMPLETE all6nodes0 actual20KiB',flush=True)


if __name__=='__main__':main()
