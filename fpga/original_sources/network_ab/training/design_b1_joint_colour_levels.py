"""Quantize chroma as well as luminance in the existing16->8->RGB tail.

Nine luminance and five levels in each opponent colour axis,225 combinations.
This tests continuous-chroma mixing independently from quadrant selection.
No inference-side histogram, palette search, sharpening or extra operation.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_quadrant_consensus import install as base_install,regression
from .design_b1_edge_stopping_diffusion import clear,KERNEL,identity_dw
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,gain=2):
    base_install(net,8,1.5)
    k='up1_low_pw';clear(net,k,.5)
    fields=[(torch.tensor([.25,.5,.25]),0,tuple(11+14*j for j in range(8))),
            (torch.tensor([.5,0.,-.5]),64,(52,60,68,76)),
            (torch.tensor([-.25,.5,-.25]),64,(52,60,68,76))]
    c=0
    for coef,bias,thresholds in fields:
        for threshold in thresholds:
            net.weight[k][c,:3,0,0]=32*coef;net.bias[k][c]=63.5+32*(bias-threshold);c+=1
    identity_dw(net,'up1_high_dw',range(16),KERNEL)
    clear(net,'up2_low_dw',1)
    net.up2_low_dw_m.fill_(131071)
    for j in range(16):net.weight['up2_low_dw'][j,0]=KERNEL*128*131071/131072;net.bias['up2_low_dw'][j]=-64*128*131071/131072
    clear(net,'up2_low_pw')
    for j in range(8):net.weight['up2_low_pw'][j,2*j:2*j+2,0,0]=.5
    clear(net,'up2_high_dw',1/32)
    for j in range(8):net.weight['up2_high_dw'][j,0]=KERNEL*gain;net.bias['up2_high_dw'][j]=64*(1-gain)
    clear(net,'rgb_head',1/512)
    # Packed pair spans0..127 for two level increments.
    net.weight['rgb_head'][:,:4,0,0]=56/127
    net.weight['rgb_head'][:,4:6,0,0]=torch.tensor([[32.],[0.],[-32.]])/127
    net.weight['rgb_head'][:,6:8,0,0]=torch.tensor([[-32.],[32.],[-32.]])/127
    net.rgb_head_m.fill_(512) # quantum1/256; largest weight~.441 fits.
    net.bias['rgb_head'].copy_(torch.tensor([-120.,-152.,-56.]))
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];assert (w/s).abs().max()<=127.001
        w.copy_(net.round(w/s)*s)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for mode,gain in (('continuous_chroma',None),('all_axes2',2),('all_axes3',3)):
        net=copy.deepcopy(base)
        if gain:install(net,gain)
        else:base_install(net,8,1.5)
        checks=regression(net,8);sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im
            records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('JOINT_COLOUR_LEVELS',mode,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('continuous_chroma','all_axes2','all_axes3')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
