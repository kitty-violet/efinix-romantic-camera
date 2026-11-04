"""Can the current B1 sharpen its OWN predicted geometry? Local control.

Project frozen full-frame predictions into connected colour regions offline,
then train only the existing decoder on untouched source RGB. The original
teacher is retained as a coarse colour/content anchor. Projection is never
called at inference. Three TRAIN scenes are only a causal diagnostic.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .whitened_b1_optimizer import WhitenedB1Adam,patch_statistics
from .minimum_area_blocks import make_target
from .b1_coherent_rgb_patch_profiles import coherent_profiles
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=2000)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100817);torch.set_num_threads(2);cv2.setNumThreads(2)
    state,net=load(a.checkpoint);net.__class__=ReadoutB1;net.cuda();data=m.load_data(a.data)
    byid={r['id']:r for r in data['training']};ss=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[ss[k] for k in (1,5,6)]
    out=Path(a.out);out.mkdir(exist_ok=False);xs=[];ts=[];qs=[];labs=[];originals=[];prep=[]
    for spec in specs:
        row=byid[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        full=m.t.r.tensor(m.t.r.read(row['path']))[None]
        with torch.no_grad():p=net(full)[0]
        image=m.t.r.c.rgb_image(p[0]);image.save(out/(row['id']+'_initial.png'))
        projected,label,desc=make_target(np.array(image),256)
        Image.fromarray(projected).save(out/(row['id']+'_self_target.png'))
        for ar,dst in ((m.t.r.read(row['path']),xs),(m.t.r.read(row['target']),ts),(projected,qs),(np.array(image),originals)):
            dst.append(torch.from_numpy(ar[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float().cuda()/128-1)
        labs.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)).cuda())
        prep.append(dict(id=row['id'],projection=desc,self_target_sha256=file_hash(out/(row['id']+'_self_target.png'))))
        print('SELF_CONTOUR_PREPARED',row['id'],desc['final720'],flush=True)
    x,t,q,old,lab=torch.stack(xs),torch.stack(ts),torch.stack(qs),torch.stack(originals),torch.stack(labs)
    cut=lambda z:z[...,40:-40,40:-40]
    # Cropped source prediction must agree with whole-frame prediction in the
    # supervised region; source crops retain the fixed stride phase.
    with torch.no_grad():assert float((cut(net(x)[0])-cut(old)).abs().max()*128)<=.51
    frozen=[];tail=False
    for row in net.plan['nodes']:
        if row['id']=='res3_add':tail=True;continue
        k=row['id'].replace('.','__')
        if k in net.weight:
            net.weight[k].requires_grad_(tail);net.bias[k].requires_grad_(tail)
            if not tail:frozen.extend(('weight.'+k,'bias.'+k))
    initial=copy.deepcopy(net.state_dict());stats=patch_statistics(net,x,4096)
    def optimizer():
        groups=[]
        for k,w in net.weight.items():
            if not w.requires_grad:continue
            s=(getattr(net,k+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=.0002,row_gain=(s*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.004)])
        return WhitenedB1Adam(groups,net,stats,True)
    opt=optimizer();params=[v for v in net.parameters() if v.requires_grad]
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,prep=prep,code=hashes,checkpoint_sha256=file_hash(a.checkpoint),
        data_sha256=file_hash(a.data),kind='frozen-source-prediction geometry self-distillation',local_fit_only=True,
        raw_source_input=True,projection_in_inference=False,frozen_parameter_keys=frozen))
    def update(step):
        opt.zero_grad();p=net(x)[0]
        pixel=F.l1_loss(cut(p),cut(q));anchor=F.l1_loss(F.avg_pool2d(cut(p),8),F.avg_pool2d(cut(t),8))
        other=net((x[:1]+(1 if step%2 else -1)/128).clamp(-1,127/128))[0]
        noise=F.smooth_l1_loss(cut(other),cut(p[:1]).detach(),beta=2/128)
        shifted=net(torch.roll(x[:1],(1,1),(-2,-1)))[0];shift=F.l1_loss(cut(p[:1]),cut(torch.roll(shifted,(-1,-1),(-2,-1))))
        loss=8*pixel+2*anchor+5*noise+1.5*shift
        loss.backward();opt.prepare_gradients();gn=torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);assert gn>0
        opt.step();assert all(torch.equal(net.state_dict()[k],initial[k]) for k in frozen)
        return float(loss.detach()),float(gn)
    update(0);torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt')
    expected=update(1);expected_state=copy.deepcopy(net.state_dict())
    disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True);net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer'])
    got=update(1);err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in expected_state.items());assert err==0 and expected==got
    net.load_state_dict(initial);opt=optimizer()
    write_json(out/'preflight.json',dict(status='PASS',gpu_full_forward=True,gpu_minibatch_backward=True,disk_next_step_error=err))
    print('SELF_CONTOUR_GPU_PREFLIGHT_PASS disk0',flush=True)
    history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);started=time.monotonic()
    for step in range(a.steps+1):
        if step%500==0 or step==a.steps:
            assert all(file_hash(p)==v for p,v in hashes.items())
            with torch.no_grad():
                nodes=net(x,all_nodes=True);actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cuda')(net.round(x*128),True)
                assert all(torch.equal(v,actual[k]) for k,v in nodes.items());p=actual['rgb_head']/128
                shape,_=coherent_profiles(cut(p),cut(t),cut(lab))
                rec=dict(step=step,mae_teacher=float((cut(p)-cut(t)).abs().mean()*128),mae_self=float((cut(p)-cut(q)).abs().mean()*128),
                    coherent_teacher=float(shape),actual_all_nodes_error=0)
            history.append(rec);write_json(out/'history.json',history)
            board=Image.new('RGB',(960,792),'#17202a');draw=ImageDraw.Draw(board)
            for i in range(3):
                for j,(title,z) in enumerate((('原固定目标',t),('初始实际B1',old),('自身轮廓收紧·仅监督',q),('实际B1·局部诊断',p))):
                    board.paste(m.t.r.c.rgb_image(cut(z)[i]),(240*j,264*i+24));draw.text((240*j+3,264*i+3),title,font=font,fill='white')
            board.save(out/f'comparison_{step:06d}.png')
            saved={k:copy.deepcopy(state[k]) for k in ('contract_sha256','residual_encoding')}
            saved.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,qat=True,local_fit_only=True,
                visual_acceptance=False,training_provenance=dict(kind='self-contour local decoder diagnostic',args=vars(a),code=hashes))
            torch.save(saved,out/f'candidate_{step:06d}.pt');print('SELF_CONTOUR_LOCAL',rec,flush=True)
        if step==a.steps:break
        update(step)
    m.t.preview(net,data['validation'],out/'whole-frame-preview')
    write_json(out/'results.json',dict(history=history,seconds=time.monotonic()-started,extra_inference_ops=0,
        local_fit_only=True,temporal_video_not_validated=True,visual_acceptance=False))


if __name__=='__main__':main()
