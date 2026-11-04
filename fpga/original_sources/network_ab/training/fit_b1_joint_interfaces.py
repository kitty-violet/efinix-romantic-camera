"""Same TRAIN source fit, change only the geometric30-field target.

Both arms use the centered optimizer, squared rank, real input pixels and same
parameters. Scores against one fixed target and actual integer replay are kept
separate from the training field loss. Local fit is not generalization.
"""
import argparse,copy,json,sys,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .b1_joint_interface_targets import joint_interface_fields,regression
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .b1_render_boundary_objective import smooth_rank_loss
from .fit_b1_distance_geometry import optimizer
from .centered_b1_optimizer import calibration_means
from .train_b1_ternary_regions import region_codes
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--device',default='cuda');ap.add_argument('--resume-joint',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026100715);regression()
    state,initial=load(a.checkpoint);initial.__class__=OrdinalBandB1;initial.to(a.device)
    dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data(a.data)
    selected=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];selected=[selected[i] for i in (1,5,6)];byid={r['id']:r for r in data['training']}
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720'];pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    xs=[];labs=[];codes=[];original=[]
    for spec in selected:
        row=byid[spec['row']['id']];assert row['sha256']==spec['row']['sha256'];xx,yy=spec['xy'];w,h=spec['wh'];sl=np.s_[yy:yy+h,xx:xx+w]
        xs.append(torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
        original.append(torch.from_numpy(m.t.r.read(row['target'])[sl].transpose(2,0,1).copy()).float()/128-1)
        labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        codes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
    x=torch.stack(xs).to(a.device);labels=torch.stack(labs).to(a.device);code=torch.stack(codes).to(a.device);target=torch.stack(original).to(a.device)
    truths={'independent':distance_fields(code,8),'joint_interface':joint_interface_fields(code)};means=calibration_means(initial,x);cut=lambda z:z[...,40:-40,40:-40]
    sources={str(Path(src).resolve()):file_hash(src) for module in list(sys.modules.values()) if (src:=getattr(module,'__file__',None)) and 'network_ab' in src and src.endswith('.py')}
    for src in sources:shutil.copy2(src,dest/Path(src).name)
    write_json(dest/'config.json',dict(args=vars(a),samples=selected,code=sources,checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),train_only=True))
    images=[x,target];records=[]
    if a.resume_joint:
        assert state.get('local_joint_interface_fit')=='joint_interface' and 'optimizer' in state
        truths={'joint_interface':truths['joint_interface']}
        with torch.no_grad():images.append(initial(x)[0])
    for mode,truth in truths.items():
        net=copy.deepcopy(initial);opt=optimizer(net,centred=True,means=means);params=[p for p in net.parameters() if p.requires_grad];history=[]
        start=0
        if a.resume_joint:opt.load_state_dict(state['optimizer']);start=state['step']
        for step in range(start,a.steps+1):
            nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128;field=nodes['up1_low_dw'][...,10:-10,10:-10]
            full,band=distance_objective(field,truth[...,10:-10,10:-10]);loss=16*full+8*band+10*smooth_rank_loss(field)
            if step%(1000 if a.resume_joint else 250)==0 or step==a.steps:
                with torch.no_grad():
                    _,v=reference_boundary_profile(cut(p),cut(target),cut(labels));row=dict(step=step,full=float(full),band=float(band),mae=float((cut(p)-cut(target)).abs().mean()*128),profile_shape=float(v['rgb_profile_shape']),profile_contrast=float(v['rgb_profile_contrast']))
                history.append(row);write_json(dest/f'{mode}_history.json',history);print('JOINT_INTERFACE_FIT',mode,row,flush=True)
                ss=copy.deepcopy(state);ss.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,local_joint_interface_fit=mode,visual_acceptance=False);torch.save(ss,dest/f'{mode}_{step:06d}.pt')
                board=Image.new('RGB',(320*(len(images)+1),1062),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
                for i in range(3):
                    for j,im in enumerate(images+[p.detach()]):
                        titles=('原图','原目标','继续前源预测','共同界面继续训练') if a.resume_joint else ('原图','原目标','独立距离训练','共同界面训练')
                        board.paste(m.t.r.c.rgb_image(im[i]),(320*j,354*i+34));draw.text((320*j+2,354*i+4),titles[j],font=font,fill='white')
                board.save(dest/f'{mode}_{step:06d}.png')
            if step==a.steps:break
            opt.zero_grad();loss.backward();opt.prepare_gradients();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=a.device);other,_=actual(net.round(x*128),True)
            assert len(raw)==20480 and all(torch.equal(v,other[k]) for k,v in nodes.items())
        images.append(p.detach());records.append(dict(mode=mode,history=history,all_nodes_error=0,bytes=20480))
        write_json(dest/'review.json',dict(records=records,train_only=True,local_fit_not_generalization=True,visual_acceptance=False))
    assert all(file_hash(p)==sha for p,sha in sources.items())
    print('JOINT_INTERFACE_FIT_COMPLETE',flush=True)


if __name__=='__main__':main()
