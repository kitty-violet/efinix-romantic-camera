"""Controlled local TRAIN fit of actual360p shared-edge alignment.

The previous180p gradient audit did not run this repair. Here the existing
post-interpolation field is aligned, and actual source predictions are checked.
This three-crop experiment is NOT evidence of generalization or video quality.
"""
import argparse,copy,json,types
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_rendered_luma_readout import RenderedLumaReadoutTraining
from .fit_b1_current_rim_priority import optimizer
from .centered_b1_optimizer import calibration_means
from .b1_joint_interface_targets import joint_interface_fields
from .b1_active_luma_interface import active_interface_loss,regression
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=500);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026100721);regression()
    state,initial=load(a.checkpoint);initial.__class__=ReadoutB1
    out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    specs=[json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'][i] for i in (1,5,6)]
    xx=[];tt=[];ll=[];cc=[]
    for spec in specs:
        row=byid[spec['row']['id']];cx,cy=spec['xy'];w,h=spec['wh'];sl=np.s_[cy:cy+h,cx:cx+w]
        def tensor(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy()).float()/128-1
        xx.append(tensor(row['path']));tt.append(tensor(row['target']));ll.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        with np.load(row['native_luma_codes']) as z:cc.append(torch.from_numpy(z['codes'][sl].astype(np.int64)))
    x=torch.stack(xx);target=torch.stack(tt);labels=torch.stack(ll);codes=torch.stack(cc)
    truth=joint_interface_fields(codes);means=calibration_means(initial,x)
    with torch.no_grad():known=fragment(initial,F.pad(initial.round(truth[:,:14]),(0,0,0,0,0,2)),'up1_low_pw')['up1_high_dw'][:,:14]
    write_json(out/'config.json',dict(args=vars(a),specs=specs,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),alignment_node='up1_high_dw',local_train_only=True))
    cut=lambda z:z[...,40:-40,40:-40]
    images=[x,target];records=[]
    for strength in (0.,16.):
        net=copy.deepcopy(initial);opt=optimizer(net,means);params=[p for p in net.parameters() if p.requires_grad];history=[]
        obj=object.__new__(RenderedLumaReadoutTraining);obj.net=net;obj.labels=labels;obj.class_labels=codes
        obj.distance_cache=lambda code,gain:truth;obj.motion_fraction=x.new_zeros(());obj.args=types.SimpleNamespace(classification=2.)
        for step in range(a.steps+1):
            nodes=net(x,all_nodes=True);obj.nodes=nodes;p=nodes['rgb_head']/128
            base,v=obj.additional_loss(cut(p),cut(target),False)
            align,fraction=active_interface_loss(nodes['up1_high_dw'][:,:14,20:-20,20:-20],known[...,20:-20,20:-20]);loss=base+strength*align
            if step%100==0 or step==a.steps:
                r=dict(step=step,loss=float(loss.detach()),mae=float((cut(p)-cut(target)).abs().mean().detach()*128),rim=float(v['two_pixel_region_centred_bright_shoulder'].detach()),alignment=float(align.detach()))
                history.append(r);print('CURRENT_INTERFACE_FIT',strength,r,flush=True)
                board=Image.new('RGB',(x.shape[-1]*(len(images)+1),(x.shape[-2]+30)*len(x)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
                for i in range(len(x)):
                    for j,z in enumerate(images+[p.detach()]):
                        board.paste(m.t.r.c.rgb_image(z[i]),(j*x.shape[-1],i*(x.shape[-2]+30)+30));draw.text((j*x.shape[-1]+2,i*(x.shape[-2]+30)+3),['TRAIN原图','目标','原目标续训','加共同边约束16'][j],font=font,fill='white')
                board.save(out/f'align{int(strength)}_{step:06d}.png');write_json(out/f'align{int(strength)}_history.json',history)
            if step==a.steps:break
            opt.zero_grad();loss.backward();opt.prepare_gradients();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');other,_=actual(net.round(x*128),True)
            assert len(raw)==20480 and all(torch.equal(nodes[k],v) for k,v in other.items())
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),step=a.steps,local_fit_only=True,alignment_weight=strength,visual_acceptance=False)
        for k in ('optimizer','rng','torch_rng','cuda_rng'):saved.pop(k,None)
        torch.save(saved,out/f'align{int(strength)}.pt');images.append(p.detach());records.append(dict(weight=strength,history=history,actual_all_nodes_error=0,bytes=20480))
        write_json(out/'review.json',dict(records=records,local_fit_not_generalization=True,visual_acceptance=False))
    print('CURRENT_INTERFACE_FIT_COMPLETE',flush=True)


if __name__=='__main__':main()
