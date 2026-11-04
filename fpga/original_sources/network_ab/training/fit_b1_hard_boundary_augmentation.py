"""Same hard TRAIN patches with/without label-guided input augmentation.

This local controlled fit is not natural-video generalization. Both arms share
the same centered optimizer, initialization and target, with no phase/flow loss.
Only the old label-guided contrast changes and their invariance loss differ.
"""
import argparse,copy,json,random,sys,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .fit_b1_distance_geometry import optimizer
from .centered_b1_optimizer import calibration_means
from .train_b1_real_region_texture import texture_intervention
from .train_b1_ternary_regions import region_codes
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .train_b1_render_boundary import decision_objective
from .b1_render_boundary_objective import smooth_rank_loss,regression as decision_regression
from .train_b1_ordered_boundary_fields import rank_loss
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--decision-compare',action='store_true')
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(2026100713)
    if a.decision_compare:decision_regression()
    state,initial=load(a.checkpoint);initial.__class__=OrdinalBandB1
    dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data(a.data)
    selected=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records']
    selected=[selected[i] for i in (1,5,6)];byid={r['id']:r for r in data['training']}
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    xs=[];labs=[];codes=[]
    for spec in selected:
        row=byid[spec['row']['id']];assert row['sha256']==spec['row']['sha256']
        xx,yy=spec['xy'];w,h=spec['wh'];sl=np.s_[yy:yy+h,xx:xx+w]
        xs.append(torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
        labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        codes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
    x=torch.stack(xs);labels=torch.stack(labs);truth=distance_fields(torch.stack(codes),8)
    means=calibration_means(initial,x);cut=lambda z:z[...,40:-40,40:-40]
    with torch.no_grad():target=fragment(initial,F.pad(initial.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
    sources={str(Path(src).resolve()):file_hash(src) for module in list(sys.modules.values())
             if (src:=getattr(module,'__file__',None)) and 'network_ab' in src and src.endswith('.py')}
    for src in sources:shutil.copy2(src,dest/Path(src).name)
    write_json(dest/'config.json',dict(args=vars(a),samples=selected,checkpoint_sha256=file_hash(a.checkpoint),
        data_sha256=file_hash(a.data),source_sha256=file_hash(__file__),code=sources,train_only=True,local_fit_not_generalization=True))
    images=[x,target];records=[]
    modes=('field_linear_rank','field_squared_rank','decision_squared_rank') if a.decision_compare else ('raw','label_guided')
    titles=('原图','已知可达目标','距离·旧顺序惩罚','距离·平方顺序惩罚','最终判定·平方顺序惩罚') if a.decision_compare else ('原图','已知可达目标','真实输入训练','标签增强训练')
    for mode in modes:
        rng=random.Random(2026100713);net=copy.deepcopy(initial);opt=optimizer(net,centred=True,means=means)
        params=[p for p in net.parameters() if p.requires_grad];history=[]
        for step in range(a.steps+1):
            if step%250==0 or step==a.steps:
                with torch.no_grad():
                    nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128
                    full,band=distance_objective(nodes['up1_low_dw'][...,10:-10,10:-10],truth[...,10:-10,10:-10])
                    profile,v=reference_boundary_profile(cut(p),cut(target),cut(labels))
                    row=dict(step=step,full=float(full),band=float(band),mae=float((cut(p)-cut(target)).abs().mean()*128),
                        profile_shape=float(v['rgb_profile_shape']),profile_contrast=float(v['rgb_profile_contrast']))
                history.append(row);write_json(dest/f'{mode}_history.json',history)
                ss=copy.deepcopy(state);ss.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,hard_boundary_local_fit=mode,visual_acceptance=False)
                torch.save(ss,dest/f'{mode}_{step:06d}.pt')
                board=Image.new('RGB',(320*(len(images)+1),354*3),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
                for i in range(3):
                    for j,im in enumerate(images+[p.detach()]):
                        title=titles[j]
                        board.paste(m.t.r.c.rgb_image(im[i]),(j*320,i*354+34));draw.text((j*320+2,i*354+4),title,font=font,fill='white')
                board.save(dest/f'{mode}_{step:06d}.png');print('HARD_BOUNDARY_AUGMENTATION',mode,row,flush=True)
            if step==a.steps:break
            xx=texture_intervention(x,labels,rng=rng) if mode=='label_guided' and step%4 in (1,2) else x
            nodes=net(xx,all_nodes=True);full,band=distance_objective(nodes['up1_low_dw'][...,10:-10,10:-10],truth[...,10:-10,10:-10])
            loss=16*full+8*band
            field=nodes['up1_low_dw'][...,10:-10,10:-10]
            if mode=='field_linear_rank':loss=loss+10*rank_loss(field)
            elif mode=='field_squared_rank':loss=loss+10*smooth_rank_loss(field)
            elif mode=='decision_squared_rank':loss=2*decision_objective(net,nodes,truth,False)[0]+10*smooth_rank_loss(field)
            if mode=='label_guided' and step%8==3:
                perturbed=texture_intervention(x[:1],labels[:1],rng=rng)
                loss=loss+2*F.smooth_l1_loss(net(perturbed)[0],nodes['rgb_head'][:1]/128,beta=4/128)
            opt.zero_grad();loss.backward();opt.prepare_gradients();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            nn=net(x,all_nodes=True);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');other,_=actual(net.round(x*128),True)
            assert len(raw)==20480 and all(torch.equal(v,other[k]) for k,v in nn.items())
        images.append(p.detach());records.append(dict(mode=mode,history=history,all_node_error=0,bytes=20480))
        write_json(dest/'review.json',dict(records=records,train_only=True,local_fit_not_generalization=True,visual_acceptance=False))
    assert all(file_hash(p)==sha for p,sha in sources.items())
    print('HARD_BOUNDARY_AUGMENTATION_COMPLETE',flush=True)


if __name__=='__main__':main()
