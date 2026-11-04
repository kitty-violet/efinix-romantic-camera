"""Separate region prediction errors from the final sampled colour decoder."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_sampling_colour_v2 as v2
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,write_json


@torch.inference_mode()
def run(package=None,out=None):
    configure_workspace();data,selected,cache=c.load_data()
    package=package or v2.OUT/'learned/qat/package/coefficients-padded.bin';out=out or v2.OUT
    out.mkdir(exist_ok=True);assert not (out/'reconstruction_audit.json').exists()
    net=c.ColourB1(False,package).cuda().eval();net.quantized=True
    board=Image.new('RGB',(1680,1248),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    def conv(x,key):
        w,b=net.weight[key],net.bias[key];m=getattr(net,key+'_m');s=(m.flatten()/131072).float()
        wi=net.round(w/s[:,None,None,None]).clamp(-127,127);bi=net.round(b/s).double()[None,:,None,None]
        if key=='up2_high_dw':x=F.pad(x,(1,1,1,1),mode='replicate')
        dot=F.conv2d(x,wi,groups=8 if key=='up2_high_dw' else 1)
        y=net.round((dot.double()+bi)*m/131072).clamp(-128,127).float()
        return y.clamp_min(0) if key=='up2_high_dw' else y
    records=[]
    for i,(key,box,label) in enumerate(c.CROPS):
        r=next(a for a in cache if a['row']['id']==key);t=v2.labels(r['target'][None]);nodes=net(r['source'][None],all_nodes=True)
        actual=nodes['rgb_head']/128;f=v2.features(net,nodes)
        # Direct target class decoding is an oracle control, not an inference model.
        gt=t.clone();gt[:,:6]/=63.5;gt[:,6:]=net.round(gt[:,6:]/2)
        direct=conv(gt,'rgb_head')/128
        lattice=t[:,:,::2,::2].repeat_interleave(2,-2).repeat_interleave(2,-1)
        sampled=conv(conv(lattice,'up2_high_dw'),'rgb_head')/128
        for j,(name,z) in enumerate([('教师原目标',r['target']),('仅诊断：13层目标直接解码',direct[0]),
                                     ('仅诊断：目标经360格点及末卷积',sampled[0]),('实际小网络',actual[0])]):
            board.paste(c.rgb_image(z).crop(box),(420*j,312*i+32));draw.text((420*j+2,312*i+4),name,font=font,fill='white')
        predicted=nodes['up2_high_dw'];stats={}
        for name in ('stem','down1.dw','down2.dw','res3_add','up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw'):
            z=nodes[name]
            stats[name]=dict(upper_saturation=float((z>=127).float().mean()),zero_fraction=float((z==0).float().mean()),
                            dormant_channels=int((z.flatten(2).std(-1)<.1).sum()))
        records.append(dict(id=key,soft_chroma_bias=(f[:,6:]-t[:,6:]).mean((0,2,3)).tolist(),
            real_chroma_bias=(predicted[:,6:]*2-t[:,6:]).mean((0,2,3)).tolist(),
            real_chroma_mae=float((predicted[:,6:]*2-t[:,6:]).abs().mean()),
            invalid_ordinal_fraction=float((predicted[:,:5]<predicted[:,1:6]).any(1).float().mean()),
            teacher_target_mae=float((direct-r['target']).abs().mean()*128),
            sampled_target_mae=float((sampled-r['target']).abs().mean()*128),
            real_mae=float((actual-r['target']).abs().mean()*128),nodes=stats))
    board.save(out/'reconstruction_oracles.png');write_json(out/'reconstruction_audit.json',records)
    for r in records:print(r['id'],'chroma_bias',r['real_chroma_bias'],'mae',r['real_chroma_mae'],'ordinal_bad',r['invalid_ordinal_fraction'],r['nodes'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    run(a.package,a.out)
