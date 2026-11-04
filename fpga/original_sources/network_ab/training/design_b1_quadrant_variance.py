"""Four-window variance selection compiled into the existing B1 graph.

A convex piecewise-linear squared-luminance surrogate replaces multiplication.
The selected window's whole RGB vector is carried through exact binary gates.
This is a four-window design, NOT the anisotropic Kuwahara paper algorithm.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

THRESH=(25,51,76,102)

@torch.no_grad()
def hard(net,key,channels):
    for c in channels:
        net.weight[key][c].zero_();net.weight[key][c,0,1,1]=127*131071/131072
        net.bias[key][c]=-64*127*131071/131072;getattr(net,key+'_m')[:,c]=131071


@torch.no_grad()
def install(net,blocks):
    for k in net.weight:clear(net,k)
    for r in range(1,4):net.package['residual_encoding'][f'res{r}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    y=torch.tensor([.25,.5,.25])
    for j,t in enumerate(THRESH):net.weight['stem'][3+j,:,1,1]=y*.5;net.bias['stem'][3+j]=64-t
    identity_dw(net,'down1__dw',range(7),KERNEL)
    # Store3*(PWL(Y^2/128)-Y)+96 in7bit. Reuse exactly these coefficients.
    q=torch.cat((3*(25/128-1)*y,torch.tensor([153,153,153,156])/128))
    q=net.round(q*64)/64
    for group in range(4):
        for c in range(3):net.weight['down1__pw'][4*group+c,c,0,0]=1
        net.weight['down1__pw'][4*group+3,:7,0,0]=q;net.bias['down1__pw'][4*group+3]=96
    identity_dw(net,'down2__dw',range(16),KERNEL)
    for j in range(16):net.weight['down2__pw'][j,j,0,0]=1
    clear(net,'res1_conv1__dw')
    for group,(dy,dx) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        for c in range(4):net.weight['res1_conv1__dw'][4*group+c,0,dy:dy+2,dx:dx+2]=.25
    for group in range(4):
        for c in range(3):net.weight['res1_conv1__pw'][3*group+c,4*group+c,0,0]=1
        net.weight['res1_conv1__pw'][12+group,4*group+3,0,0]=1
        for j,t in enumerate(THRESH):
            net.weight['res1_conv1__pw'][16+4*group+j,4*group:4*group+3,0,0]=y
            net.bias['res1_conv1__pw'][16+4*group+j]=-t
    identity_dw(net,'res1_conv2__dw',range(32))
    variance=torch.zeros(4,32)
    for group in range(4):
        variance[group,12+group]=1;variance[group,3*group:3*group+3]=-q[:3]
        variance[group,16+4*group:20+4*group]=-q[3:]
    k='res1_conv2__pw';clear(net,k,1/32)
    for j in range(12):net.weight[k][j,j,0,0]=1
    net.weight[k][12:16,:,0,0]=variance;net.bias[k][12:16]=-96
    for j in range(2):net.weight[k][16+j,:,0,0]=variance[2*j]-variance[2*j+1];net.bias[k][16+j]=64
    identity_dw(net,'res2_conv1__dw',range(18));hard(net,'res2_conv1__dw',(16,17))
    # Two pairwise tournaments. Each mask acts on a whole RGB vector and V.
    for group in range(4):
        gate=16+group//2;sgn=1 if group%2 else -1
        for c in range(3):
            j=3*group+c;net.weight['res2_conv1__pw'][j,j,0,0]=1
            net.weight['res2_conv1__pw'][j,gate,0,0]=sgn;net.bias['res2_conv1__pw'][j]=-127 if sgn==1 else 0
        j=12+group;net.weight['res2_conv1__pw'][j,j,0,0]=1;net.weight['res2_conv1__pw'][j,gate,0,0]=sgn;net.bias['res2_conv1__pw'][j]=-127 if sgn==1 else 0
    identity_dw(net,'res2_conv2__dw',range(16))
    k='res2_conv2__pw'
    for group in range(2):
        for c in range(3):net.weight[k][3*group+c,6*group+c,0,0]=1;net.weight[k][3*group+c,6*group+3+c,0,0]=1
        net.weight[k][6+group,12+2*group:14+2*group,0,0]=1
    net.weight[k][8,12:16,0,0]=torch.tensor([1.,1,-1,-1]);net.bias[k][8]=64
    identity_dw(net,'res3_conv1__dw',range(9));hard(net,'res3_conv1__dw',(8,))
    k='res3_conv1__pw'
    for j in range(6):net.weight[k][j,j,0,0]=1;net.weight[k][j,8,0,0]=-1 if j<3 else 1;net.bias[k][j]=0 if j<3 else -127
    identity_dw(net,'res3_conv2__dw',range(6))
    for j in range(3):net.weight['res3_conv2__pw'][j,j,0,0]=1;net.weight['res3_conv2__pw'][j,j+3,0,0]=1
    identity_dw(net,'up1_low_dw',range(3))
    if blocks:
        k='up1_low_pw';clear(net,k,.5)
        for j in range(14):net.weight[k][j,:3,0,0]=32*y;net.bias[k][j]=63.5-32*(8+8*j)
        for j,coef in ((14,torch.tensor([.5,0,-.5])),(15,torch.tensor([-.25,.5,-.25]))):
            getattr(net,k+'_m')[:,j]=2048;net.weight[k][j,:3,0,0]=coef;net.bias[k][j]=64
        identity_dw(net,'up1_high_dw',range(16),KERNEL)
        identity_dw(net,'up2_low_dw',range(16));hard(net,'up2_low_dw',range(14))
        clear(net,'up2_low_pw',1/32);net.up2_low_pw_m[:,0]=8257;net.weight['up2_low_pw'][0,:14,0,0]=8257/131072
        net.weight['up2_low_pw'][1,14,0,0]=1;net.weight['up2_low_pw'][2,15,0,0]=1
        identity_dw(net,'up2_high_dw',range(3),KERNEL)
        clear(net,'rgb_head',1/32);net.weight['rgb_head'][:,:3,0,0]=torch.tensor([[2,2,-2],[2,0,2],[2,-2,-2]])
        net.bias['rgb_head'].copy_(torch.tensor([-120.,-248.,136.]))
    else:
        for j in range(3):net.weight['up1_low_pw'][j,j,0,0]=1
        identity_dw(net,'up1_high_dw',range(3),KERNEL*.25+torch.tensor([[0.,0,0],[0,.75,0],[0,0,0]]))
        identity_dw(net,'up2_low_dw',range(3))
        for j in range(3):net.weight['up2_low_pw'][j,j,0,0]=1
        identity_dw(net,'up2_high_dw',range(3),KERNEL)
        clear(net,'rgb_head',1/32)
        for j in range(3):net.weight['rgb_head'][j,j,0,0]=2
        net.bias['rgb_head'].fill_(-128)
    # Quantize once, then all references below use the actual coefficients.
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert (w/s).abs().max()<=127.001,(k,float((w/s).abs().max()))
        w.copy_(net.round(w/s)*s)


@torch.no_grad()
def regression(net):
    # Test the selector with arbitrary legal means and nonnegative variance.
    torch.manual_seed(817);z=torch.zeros(64,32,6,8)
    rgb=torch.randint(0,128,(64,4,3,6,8)).float();v=torch.randint(0,80,(64,4,6,8)).float()
    z[:,:12]=rgb.flatten(1,2);z[:,12:16]=v
    z[:,16]=64+v[:,0]-v[:,1];z[:,17]=64+v[:,2]-v[:,3]
    z=z.clamp(-128,127)
    got=fragment(net,z,'res1_add')['res3_add'][:,:3]
    choice=v.argmin(1);want=rgb.gather(1,choice[:,None,None].expand(-1,1,3,-1,-1)).squeeze(1)
    assert torch.equal(got,want),float((got-want).abs().max())
    return dict(random_common_rgb_min_variance_selector_error=0,tie_break_first=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for blocks in (False,True):
        net=copy.deepcopy(base);install(net,blocks);checks=regression(net);name='levels' if blocks else 'rgb';sub=out/name;sub.mkdir()
        raw=net.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw)
        rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[name,row['id']]=im
            records.append(dict(variant=name,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],
            designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('QUADRANT_VARIANCE',name,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('原片',Image.open(rows[rid]['path'])),('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('rgb','levels')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),
        designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0,not_anisotropic_kuwahara=True,
        reference='https://www.kyprianidis.com/p/pg2009/'))

if __name__=='__main__':main()
