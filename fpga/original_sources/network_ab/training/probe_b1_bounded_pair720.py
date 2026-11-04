"""Late720 two-colour switch with a componentwise range guarantee.

F(A,B,g)=relu(A-g)+g-relu(g-B), 0<=A,B,g<=127.
F(A,B,0)=A; F(A,B,127)=B; F(A,A,g)=A in exact arithmetic.
Seven existing channels implement it. Integer rounding is checked separately.
The RGB path is not generally a straight segment: no such claim is made.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_edge_stopping_diffusion import clear,KERNEL,identity_dw
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,gain=4):
    identity_dw(net,'up2_low_dw',range(7))
    clear(net,'up2_low_dw',1/8 if gain>=8 else 1/16)
    for c in range(6):net.weight['up2_low_dw'][c,0,1,1]=1
    net.weight['up2_low_dw'][6,0,1,1]=gain;net.bias['up2_low_dw'][6]=64*(1-gain)
    clear(net,'up2_low_pw')
    for c in range(3):
        net.weight['up2_low_pw'][c,c,0,0]=.5;net.weight['up2_low_pw'][c,6,0,0]=-.5;net.bias['up2_low_pw'][c]=64
        net.weight['up2_low_pw'][c+3,c+3,0,0]=-.5;net.weight['up2_low_pw'][c+3,6,0,0]=.5;net.bias['up2_low_pw'][c+3]=64
    net.weight['up2_low_pw'][6,6,0,0]=1
    clear(net,'up2_high_dw',1/32)
    for c in range(6):net.weight['up2_high_dw'][c,0]=2*KERNEL;net.bias['up2_high_dw'][c]=-128
    net.weight['up2_high_dw'][6,0]=KERNEL
    clear(net,'rgb_head',1/32)
    for c in range(3):
        net.weight['rgb_head'][c,c,0,0]=2;net.weight['rgb_head'][c,c+3,0,0]=-2;net.weight['rgb_head'][c,6,0,0]=2
    net.bias['rgb_head'].fill_(-128)
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert (w/s).abs().max()<=127.001,(k,float((w/s).abs().max()))

@torch.no_grad()
def regression(net,gain):
    torch.manual_seed(829)
    # Random RGB endpoints; all128 gates, including former chromatic counterexample.
    z=torch.zeros(256,16,1,128);z[:,:6]=torch.randint(0,128,(256,6,1,1));z[:,6]=torch.arange(128)
    z[0,:6,0]=torch.tensor([30,90,30,90,30,30])[:,None]
    z[128:,3:6]=z[128:,:3]
    p=fragment(net,z,'up1_high_dw')['rgb_head'];a=z[:,:3].repeat_interleave(2,-2).repeat_interleave(2,-1)*2-128;b=z[:,3:6].repeat_interleave(2,-2).repeat_interleave(2,-1)*2-128
    bound=torch.maximum((torch.minimum(a,b)-p).clamp_min(0),(p-torch.maximum(a,b)).clamp_min(0)).max()
    constant=(p[128:]-a[128:]).abs().max();assert bound<=4 and constant<=4,(float(bound),float(constant))
    # Exact scalar algebra for every integer pair and gate, independent of graph.
    aa=torch.arange(128)[:,None,None];bb=torch.arange(128)[None,:,None];gg=torch.arange(128)[None,None,:]
    f=(aa-gg).clamp_min(0)+gg-(gg-bb).clamp_min(0)
    assert ((f>=torch.minimum(aa,bb))&(f<=torch.maximum(aa,bb))).all()
    assert torch.equal(f[:,:,0],aa[:,:,0].expand(128,128)) and torch.equal(f[:,:,-1],bb[:,:,0].expand(128,128))
    return dict(exhaustive_scalar_cases=128**3,range_violation=0,integer_rgb_range_excess=float(bound),integer_constant_error=float(constant),not_rgb_convex_segment=True)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    assert state['two_colour_fields']
    for mode,gain in (('old',None),('bounded4',4),('bounded8',8)):
        net=copy.deepcopy(base)
        if gain:install(net,gain)
        checks=regression(net,gain) if gain else {};sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im
            records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),bounded_pair720=gain,visual_acceptance=False);saved.pop('optimizer',None);torch.save(saved,sub/'candidate.pt')
        print('BOUNDED_PAIR720',mode,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('old','bounded4','bounded8')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
