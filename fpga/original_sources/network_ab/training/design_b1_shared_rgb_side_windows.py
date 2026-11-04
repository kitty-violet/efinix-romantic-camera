"""Two common-RGB side-window decisions compiled into unchanged B1.

Independent coefficient construction inspired by the side-window principle,
not an implementation of the paper's eight-window algorithm. The actual
three-colour vector is selected by ONE L1 decision, not three scalar medians.
All gates are existing clipped pointwise convolutions; no inference helper.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def windows(vertical=False):
    a=torch.zeros(3,3);b=a.clone();a[1,1]=b[1,1]=.5
    a[0 if vertical else 1,1 if vertical else 0]=.5
    b[2 if vertical else 1,1 if vertical else 2]=.5
    eye=torch.zeros(3,3);eye[1,1]=1
    return a,b,eye


@torch.no_grad()
def absolute_pw(net,key,paired=False):
    clear(net,key)
    def terms(which,c):
        return [(which*6+c,1),(which*6+3+c,1)] if paired else [(which*3+c,1)]
    for side in range(2):
        for c in range(3):
            for sign in range(2):
                o=side*6+2*c+sign;sgn=1 if sign==0 else -1
                for i,v in terms(side,c):net.weight[key][o,i,0,0]+=sgn*v
                for i,v in terms(2,c):net.weight[key][o,i,0,0]-=sgn*v
            for i,v in terms(side,c):net.weight[key][12+side*3+c,i,0,0]=v


@torch.no_grad()
def gate_pw(net,key):
    clear(net,key);scale=131071/131072
    getattr(net,key+'_m')[:,0]=131071
    net.weight[key][0,:6,0,0]=-127*scale
    net.weight[key][0,6:12,0,0]=127*scale
    net.bias[key][0]=127*scale
    for c in range(6):net.weight[key][1+c,12+c,0,0]=1


@torch.no_grad()
def select_pw(net,key,copies=1):
    clear(net,key)
    for rep in range(copies):
        for c in range(3):
            for side in range(2):
                o=rep*6+side*3+c
                net.weight[key][o,1+side*3+c,0,0]=1
                net.weight[key][o,0,0,0]=1 if side==0 else -1
                net.bias[key][o]=-127 if side==0 else 0


@torch.no_grad()
def install(net,mix=.25):
    for k in net.weight:clear(net,k)
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    box=torch.zeros(3,3);box[1:,1:]=.25
    identity_dw(net,'down1__dw',range(3),box)
    for c in range(3):net.weight['down1__pw'][c,c,0,0]=1
    identity_dw(net,'down2__dw',range(3),box)
    for group in range(3):
        for c in range(3):net.weight['down2__pw'][group*3+c,c,0,0]=1
    for group,kernel in enumerate(windows()):net.weight['res1_conv1__dw'][group*3:group*3+3,0]=kernel
    absolute_pw(net,'res1_conv1__pw')
    identity_dw(net,'res1_conv2__dw',range(18));gate_pw(net,'res1_conv2__pw')
    identity_dw(net,'res2_conv1__dw',range(7));select_pw(net,'res2_conv1__pw',3)
    for group,kernel in enumerate(windows(True)):net.weight['res2_conv2__dw'][group*6:group*6+6,0]=kernel
    absolute_pw(net,'res2_conv2__pw',True)
    identity_dw(net,'res3_conv1__dw',range(18));gate_pw(net,'res3_conv1__pw')
    identity_dw(net,'res3_conv2__dw',range(7));select_pw(net,'res3_conv2__pw')
    identity_dw(net,'up1_low_dw',range(6))
    for c in range(3):net.weight['up1_low_pw'][c,c,0,0]=net.weight['up1_low_pw'][c,c+3,0,0]=1
    kernel=KERNEL*mix;kernel[1,1]+=1-mix
    identity_dw(net,'up1_high_dw',range(3),kernel)
    identity_dw(net,'up2_low_dw',range(3))
    for c in range(3):net.weight['up2_low_pw'][c,c,0,0]=1
    identity_dw(net,'up2_high_dw',range(3),KERNEL)
    clear(net,'rgb_head',1/32)
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2
    net.bias['rgb_head'].fill_(-128)
    for r in range(1,4):net.package['residual_encoding'][f'res{r}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    net.quantized=True
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert torch.equal(w,net.round(w/s)*s) and (w/s).abs().max()<=127,(k,'lattice')


def filtered(x,k):
    return m.t.r.c.Round()(F.conv2d(F.pad(x,(1,)*4,mode='replicate'),k[None,None].expand(x.shape[1],1,3,3),groups=x.shape[1]))


def choose(left,right,centre):
    gate=((left-centre).abs().sum(1,keepdim=True)<=(right-centre).abs().sum(1,keepdim=True))
    return torch.cat((left.clamp(0,127)*gate,right.clamp(0,127)*~gate),1),gate


@torch.no_grad()
def regression(net):
    torch.manual_seed(82682);rgb=torch.randint(0,128,(4,3,32,36)).float();z=torch.zeros(4,32,32,36)
    for group in range(3):z[:,group*3:group*3+3]=rgb
    part=fragment(net,z,'down2.pw');a,b,_=windows();pair,gate=choose(filtered(rgb,a),filtered(rgb,b),rgb)
    assert torch.equal(part['res2_conv1.pw'][:,:6],pair)
    a,b,_=windows(True)
    sums=lambda p:(p[:,:3]+p[:,3:]).clamp(0,127)
    rawsum=lambda p:p[:,:3]+p[:,3:]
    # The PW computes absolute differences before its clipped carry outputs;
    # two independently rounded half branches can sum to128 at a gate change.
    pair2,gate2=choose(rawsum(filtered(pair,a)),rawsum(filtered(pair,b)),rawsum(pair))
    # Residual terminal PW is LINEAR in B1; its following identity DW
    # supplies the required ReLU. Compare after that real activation.
    assert torch.equal(part['up1_low_dw'][:,:6],pair2)
    assert torch.equal(part['up1_low_pw'][:,:3],sums(pair2))
    assert set(part['res2_conv1.dw'][:,0].unique().tolist())<=set((0.,127.))
    assert set(part['res3_conv1.pw'][:,0].unique().tolist())<=set((0.,127.))
    # Common gate can only choose the complete candidate RGB vector.
    for level in ((0,0,0),(13,69,122),(127,127,127)):
        constant=torch.tensor(level).float()[None,:,None,None].expand_as(rgb)
        pp,_=choose(filtered(constant,windows()[0]),filtered(constant,windows()[1]),constant)
        assert torch.equal(sums(pp),constant)
    return dict(actual_two_shared_rgb_gates_error=0,constant_error=0,gates_binary=True,
                vertical_half_rounding_per_masked_branch=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);data=m.load_data(a.data);pictures={};records=[]
    for label,mix in (('tight',0.),('quarter',.25),('smooth',1.)):
        net=copy.deepcopy(base);install(net,mix);checks=regression(net);sub=out/label;sub.mkdir()
        raw=net.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw)
        rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));pictures[label,row['id']]=im
            records.append(dict(variant=label,id=row['id'],actual_all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            contract_sha256=state['contract_sha256'],step=0,qat=False,designed_not_trained=True,
            visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('SHARED_SIDE_WINDOW',label,checks,flush=True)
    rows={r['id']:r for r in data['validation']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    board=Image.new('RGB',(2100,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('原片',Image.open(rows[rid]['path'])),('原目标',Image.open(rows[rid]['target']))]+[(key,pictures[key,rid]) for key in ('tight','quarter','smooth')]
        for j,(title,im) in enumerate(ims):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+3),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,code_sha256=file_hash(__file__),
        design='two directional common-RGB side selections; not full SWF',visual_acceptance=False,extra_inference_ops=0))


if __name__=='__main__':main()
