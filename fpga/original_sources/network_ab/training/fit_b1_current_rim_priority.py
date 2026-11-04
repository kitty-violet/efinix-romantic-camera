"""Controlled TRAIN-only rim-priority fit in the current direct14 network.

Both arms have identical input crops, weights, decoder and centred updates.
Only the existing region-centred white-shoulder weight changes from1 to4.
Local fit is a diagnosis, never a claim of generalization or a release.
"""
import argparse,copy,json,sys,types
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1,TrainableLumaReadoutTraining
from .train_b1_luma_chroma_interfaces import luminance_codes
from .b1_joint_interface_targets import joint_interface_fields
from .centered_b1_optimizer import CenteredB1Adam,calibration_means
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def optimizer(net,means):
    groups=[]
    for k,w in net.weight.items():
        live=k not in ('down1__dw','down2__dw') and not k.startswith(('up1_high','up2_','rgb_head'))
        w.requires_grad_(live);net.bias[k].requires_grad_(live)
        if live:
            scale=(getattr(net,k+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=.0001,row_gain=(scale*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.008)])
    return CenteredB1Adam(groups,net,means)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--device',default='cpu');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026100719);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state,initial=load(a.checkpoint);assert state.get('trainable_luma_readout');initial.__class__=ReadoutB1;initial.to(a.device)
    dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];selected=[mined[i] for i in (1,5,6)]
    xx=[];tt=[];ll=[]
    for spec in selected:
        row=byid[spec['row']['id']];assert row['sha256']==spec['row']['sha256'];cx,cy=spec['xy'];w,h=spec['wh'];sl=np.s_[cy:cy+h,cx:cx+w]
        tensor=lambda path:torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy()).float()/128-1
        xx.append(tensor(row['path']));tt.append(tensor(row['target']));ll.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
    x=torch.stack(xx).to(a.device);target=torch.stack(tt).to(a.device);labels=torch.stack(ll).to(a.device)
    code=luminance_codes(target,labels,initial.round);truth=joint_interface_fields(code);means=calibration_means(initial,x)
    sources={str(Path(src).resolve()):file_hash(src) for mod in list(sys.modules.values()) if (src:=getattr(mod,'__file__',None)) and 'network_ab' in src and src.endswith('.py')}
    write_json(dest/'config.json',dict(args=vars(a),selected=selected,sources=sources,train_only=True,checkpoint_sha256=file_hash(a.checkpoint)))
    cut=lambda z:z[...,40:-40,40:-40]
    images=[x.detach(),target.detach()];records=[]
    for strength in (1.,4.):
        net=copy.deepcopy(initial);opt=optimizer(net,means);params=[p for p in net.parameters() if p.requires_grad];history=[]
        obj=object.__new__(TrainableLumaReadoutTraining);obj.net=net;obj.labels=labels;obj.class_labels=code
        obj.distance_cache=lambda code,gain:truth;obj.motion_fraction=x.new_zeros(());obj.args=types.SimpleNamespace(classification=2.)
        for step in range(a.steps+1):
            nodes=net(x,all_nodes=True);obj.nodes=nodes;p=nodes['rgb_head']/128
            loss,v=obj.additional_loss(cut(p),cut(target),False);rim=v['two_pixel_region_centred_bright_shoulder'];loss=loss+(strength-1)*rim
            if step%100==0 or step==a.steps:
                with torch.no_grad():
                    _,profile=reference_boundary_profile(cut(p),cut(target),cut(labels))
                    r=dict(step=step,loss=float(loss.detach()),mae=float((cut(p)-cut(target)).abs().mean()),rim=float(rim.detach()),
                           profile_shape=float(profile['rgb_profile_shape']),profile_contrast=float(profile['rgb_profile_contrast']))
                r['mae']*=128;history.append(r);print('CURRENT_RIM_FIT',strength,r,flush=True)
                board=Image.new('RGB',(x.shape[-1]*(len(images)+1),(x.shape[-2]+34)*len(x)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
                for i in range(len(x)):
                    for j,z in enumerate(images+[p.detach()]):
                        names=['TRAIN原图','原风格目标','白肩权重1','白肩权重4']
                        board.paste(m.t.r.c.rgb_image(z[i]),(j*x.shape[-1],i*(x.shape[-2]+34)+34));draw.text((j*x.shape[-1]+2,i*(x.shape[-2]+34)+4),names[j],font=font,fill='white')
                board.save(dest/f'rim{int(strength)}_{step:06d}.png');write_json(dest/f'rim{int(strength)}_history.json',history)
            if step==a.steps:break
            opt.zero_grad();loss.backward();opt.prepare_gradients();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=a.device);other,_=actual(net.round(x*128),True)
            assert len(raw)==20480 and all(torch.equal(nodes[k],v) for k,v in other.items())
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),optimizer=opt.state_dict(),step=a.steps,local_rim_weight=strength,local_fit_only=True,visual_acceptance=False)
        torch.save(saved,dest/f'rim{int(strength)}.pt');images.append(p.detach());records.append(dict(weight=strength,history=history,actual_all_nodes_error=0,bytes=20480))
        write_json(dest/'review.json',dict(records=records,local_fit_not_generalization=True,visual_acceptance=False))
    assert all(file_hash(p)==h for p,h in sources.items());print('CURRENT_RIM_FIT_COMPLETE',flush=True)


if __name__=='__main__':main()
