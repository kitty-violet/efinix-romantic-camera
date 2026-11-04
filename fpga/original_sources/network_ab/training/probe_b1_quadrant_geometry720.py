"""Keep region choices; compare geometry smoothing and late720 level ramps."""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_quadrant_variance import install as base_install,regression
from .design_b1_edge_stopping_diffusion import KERNEL,clear
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,mode):
    base_install(net,True)
    if mode=='base':return
    n=12 if mode=='late720' else 14
    for j in range(n):
        net.weight['up2_low_dw'][j,0]=KERNEL*128*131071/131072
        net.bias['up2_low_dw'][j]=-64*128*131071/131072
    if mode=='late720':
        net.weight['up1_low_pw'][:14].zero_();net.bias['up1_low_pw'][:14].zero_()
        for j in range(12):net.weight['up1_low_pw'][j,:3,0,0]=torch.tensor([8.,16,8]);net.bias['up1_low_pw'][j]=63.5-32*(9+10*j)
        clear(net,'up2_low_pw',1/64)
        for j in range(6):net.weight['up2_low_pw'][j,2*j:2*j+2,0,0]=.5
        for j in (6,7):net.weight['up2_low_pw'][j,j+8,0,0]=1
        clear(net,'up2_high_dw',1/64)
        for j in range(8):net.weight['up2_high_dw'][j,0]=KERNEL*(2 if j<6 else 1);net.bias['up2_high_dw'][j]=-64 if j<6 else 0
        clear(net,'rgb_head',1/32);net.weight['rgb_head'][:,:6,0,0]=.3125
        net.weight['rgb_head'][:,6:8,0,0]=torch.tensor([[2.,-2],[0,2],[-2,-2]])
        net.bias['rgb_head'].copy_(torch.tensor([-120.,-248.,136.]))
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];assert (w/s).abs().max()<=127.001
        w.copy_(net.round(w/s)*s)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for mode in ('base','smooth360','late720'):
        net=copy.deepcopy(base);install(net,mode);checks=regression(net);sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        grey=torch.arange(128);z=torch.zeros(128,32,1,1);z[:,:3]=grey[:,None,None,None]
        pp=fragment(net,z,'res3_add')['rgb_head'][:,:,0,0]
        assert torch.equal(pp[:,0],pp[:,1]) and torch.equal(pp[:,0],pp[:,2]) and (torch.diff(pp[:,0])>=0).all()
        checks.update(neutral_axis_exact=True,grey_levels=len(pp[:,0].unique()))
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im
            records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('QUADRANT_GEOMETRY',mode,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('base','smooth360','late720')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
