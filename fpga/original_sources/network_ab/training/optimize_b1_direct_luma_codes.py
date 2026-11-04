"""Invert the fixed direct14 decoder for geometry targets, TRAIN only.

Unlike the earlier30-field and two-colour probes this tests the current14Y
representation on the same three reviewed training crops. Optimized codes
are training labels, not source inference or deployable style weights.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from . import train_b1_minimum_area as m


def decode(net,z,proxy):
    result=fragment(net,F.pad(net.round(z).clamp(0,127),(0,0,0,0,0,2)),'up1_low_pw')
    decision=result['up2_low_dw']
    if proxy:
        pre=11*result['up1_high_dw'][:,:14]-635
        clipped=pre.clamp(0,127)
        value=ExactValueProxyGradient.apply(decision[:,:14],clipped+.05*(pre-clipped))
        decision=torch.cat((value,decision[:,14:]),1)
    return fragment(net,decision,'up2_low_dw')['up2_high_dw'][:,:1]/64+.125


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100731);torch.set_num_threads(2)
    _,net=load(a.checkpoint);net.cuda().requires_grad_(False);raw=net.export();out=Path(a.out);out.mkdir(exist_ok=False)
    d=m.load_data(a.data);byid={r['id']:r for r in d['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[mined[i] for i in (1,5,6)]
    codes=[];labs=[]
    for sp in specs:
        row=byid[sp['row']['id']];xx,yy=sp['xy'];w,h=sp['wh'];sl=np.s_[yy:yy+h,xx:xx+w]
        with np.load(row['native_luma_codes']) as z:codes.append(torch.from_numpy(z['codes'][sl].astype(np.int64)))
        labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
    code=torch.stack(codes).cuda();labels=torch.stack(labs).cuda();target=((code//81*16+16).float()/128)[:,None]
    initial=joint_interface_fields(code)[:,:14];z=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([z],lr=.5)
    with torch.no_grad():before=decode(net,z,False);assert torch.equal(before,decode(net,z,True))
    cut=lambda x:x[...,16:-16,16:-16];history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for step in range(a.steps+1):
        p=decode(net,z,True);pp,tt=cut(p),cut(target)
        profile,v=reference_boundary_profile(pp.expand(-1,3,-1,-1),tt.expand(-1,3,-1,-1),cut(labels))
        pixel=F.smooth_l1_loss(pp,tt,beta=2/128)
        proximity=((z-initial)/127).square().mean()
        loss=4*pixel+2*profile+.01*proximity
        if step%200==0 or step==a.steps:
            r=dict(step=step,loss=float(loss.detach()),mae=float((pp-tt).abs().mean().detach()*128),profile=float(profile.detach()));history.append(r);print('DIRECT14_TARGET_INVERSE',r,flush=True)
            board=Image.new('RGB',(960,1050),'#17202a');draw=ImageDraw.Draw(board)
            for i in range(3):
                for j,(v,name) in enumerate(((target,'目标原生亮度'),(before,'旧解析标签·非预测'),(p.detach(),'逆解标签·非预测'))):
                    im=Image.fromarray((v[i,0]*128).clamp(0,255).byte().cpu().numpy(),'L').convert('RGB');board.paste(im,(j*320,i*350+30));draw.text((j*320+3,i*350+3),name,font=font,fill='white')
            board.save(out/f'inverse_{step:06d}.png');write_json(out/'history.json',history)
        if step==a.steps:break
        opt.zero_grad();loss.backward();assert torch.isfinite(z.grad).all();opt.step()
        with torch.no_grad():z.copy_(z.clamp(0,127).sort(1,descending=True).values)
    with torch.no_grad():assert net.export()==raw and torch.equal(decode(net,z,False),decode(net,z,True))
    torch.save(dict(codes=net.round(z).cpu(),specs=specs,history=history,training_only=True,not_source_prediction=True),out/'optimized_targets.pt')
    write_json(out/'review.json',dict(history=history,coefficient_bytes_unchanged=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),target_optimization_not_network_training=True,visual_acceptance=False))


if __name__=='__main__':main()
