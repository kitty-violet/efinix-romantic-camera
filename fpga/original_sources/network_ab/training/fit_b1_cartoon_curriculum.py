"""Three-TRAIN-crop continuation from cartoon reconstruction to real photos.

Only training inputs follow the declared homotopy. Every preview and integer
replay uses untouched real source RGB. This is a local learning test, not a
general-purpose model or a demonstration with target-assisted inference.
"""
import argparse,copy,json,shutil,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .whitened_b1_optimizer import WhitenedB1Adam,patch_statistics
from .b1_geometry_first_objective import geometry_first
from .b1_relative_rgb_profiles import relative_profiles
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--blend-steps',type=int,default=1600)
    ap.add_argument('--interval',type=int,default=400);ap.add_argument('--lr',type=float,default=.0002)
    ap.add_argument('--relative-weight',type=float,default=0.)
    a=ap.parse_args();assert 0<=a.blend_steps<a.steps
    configure_workspace();set_reproducible(2026100802);torch.set_num_threads(2)
    state,initial=load(a.checkpoint);initial.__class__=ReadoutB1;initial.cuda().requires_grad_(True)
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[mined[i] for i in (1,5,6)]
    xs=[];ts=[];ls=[]
    for spec in specs:
        row=byid[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        def read(k):return torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float().cuda()/128-1
        xs.append(read('path'));ts.append(read('target'))
        ls.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)).cuda())
    x,target,labels=torch.stack(xs),torch.stack(ts),torch.stack(ls);cut=lambda z:z[...,40:-40,40:-40]
    with torch.no_grad():baseline=initial(x)[0];stats=patch_statistics(initial,x,4096)
    def optimizer(net):
        groups=[]
        for k,w in net.weight.items():
            scale=(getattr(net,k+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=a.lr,row_gain=(scale*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.008)])
        return WhitenedB1Adam(groups,net,stats,True)
    def objective(net,source_fraction=1.):
        # Fraction1 must be byte-for-byte the real source, not a regenerated
        # blend or an offline target-derived correction at inference.
        inp=x if source_fraction==1. else target.lerp(x,source_fraction)
        nodes=net(inp,all_nodes=True);p=nodes['rgb_head']/128
        loss,terms=geometry_first(cut(p),cut(target),cut(labels))
        if a.relative_weight:
            relative,details=relative_profiles(cut(p),cut(target),cut(labels))
            loss=loss+a.relative_weight*relative;terms.update(details)
        return loss,terms,nodes,p
    def update(net,opt,fraction):
        opt.zero_grad();loss,_,_,_=objective(net,fraction);loss.backward();opt.prepare_gradients()
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);assert norm>0
        opt.step();return float(loss.detach()),float(norm)
    def integer_check(net):
        with torch.no_grad():
            nodes=net(x,all_nodes=True)
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cuda')(net.round(x*128),True)
            errors={k:float((v-actual[k]).abs().max()) for k,v in nodes.items()}
        assert max(errors.values())==0
    out=Path(a.out);out.mkdir(exist_ok=False)
    paths=sorted({Path(mod.__file__).resolve() for name,mod in tuple(sys.modules.items()) if name.startswith('network_ab.training') and getattr(mod,'__file__',None)}|{Path(__file__).resolve()})
    frozen={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),code=frozen,specs=specs,checkpoint_sha256=file_hash(a.checkpoint),
        data_sha256=file_hash(a.data),training_input='untouched real source' if a.blend_steps==0 else 'target.lerp(source,min(step/blend_steps,1))',
        preview_and_final_input='untouched real source',local_fit_not_generalization=True,extra_inference_ops=0))
    # Real GPU updates for photo and mixed inputs; actual disk next-step check.
    net=copy.deepcopy(initial);opt=optimizer(net);first=update(net,opt,1.);mixed=update(net,opt,.5)
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'preflight.pt')
    expected=update(net,opt,1.)
    other=copy.deepcopy(initial);op=optimizer(other);disk=torch.load(out/'preflight.pt',map_location='cuda',weights_only=True)
    other.load_state_dict(disk['model']);op.load_state_dict(disk['optimizer']);got=update(other,op,1.)
    err=max(float((v-other.state_dict()[k]).abs().max()) for k,v in net.state_dict().items())
    assert err==0 and got==expected;integer_check(net)
    write_json(out/'preflight.json',dict(status='PASS',source=first,mixed=mixed,next_update=expected,disk_resume_error=err,all_nodes_error=0))
    print('CARTOON_CURRICULUM_PREFLIGHT_PASS GPU_source/mixed/backward/disk_resume0/all_nodes0',flush=True)
    del net,opt,other,op,disk;torch.cuda.empty_cache()
    net=copy.deepcopy(initial);opt=optimizer(net);history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);started=time.monotonic()
    for step in range(a.steps+1):
        fraction=min(step/a.blend_steps,1.) if a.blend_steps else 1.
        if step%a.interval==0 or step==a.steps:
            assert all(file_hash(p)==v for p,v in frozen.items())
            with torch.no_grad():
                loss,terms,nodes,p=objective(net,1.);self_p=net(target)[0]
                r=dict(step=step,training_source_fraction=fraction,source_rgb_mae=float((cut(p)-cut(target)).abs().mean()*128),
                    target_input_rgb_mae=float((cut(self_p)-cut(target)).abs().mean()*128),source_loss=float(loss),terms={k:float(v) for k,v in terms.items()})
            integer_check(net)
            board=Image.new('RGB',(1280,1044),'#17202a');draw=ImageDraw.Draw(board)
            for i in range(3):
                for j,(title,values) in enumerate((('固定目标',target),('初始真实源预测',baseline),('真实源预测·仅局部训练',p),('目标输入重建·仅诊断',self_p))):
                    board.paste(m.t.r.c.rgb_image(values[i]),(320*j,348*i+28));draw.text((320*j+3,348*i+3),title,font=font,fill='white')
            board.save(out/f'source_{step:06d}.png')
            saved=copy.deepcopy(state);saved.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,
                curriculum_source_fraction=fraction,local_fit_only=True,visual_acceptance=False,target_input_diagnostic=False,
                extra_inference_ops=0,real_source_preview=True)
            torch.save(saved,out/f'candidate_{step:06d}.pt');history.append(r);write_json(out/'history.json',history)
            print('CARTOON_CURRICULUM_SOURCE',r,flush=True)
        if step==a.steps:break
        update(net,opt,fraction)
    write_json(out/'review.json',dict(history=history,seconds=time.monotonic()-started,all_nodes_error=0,
        last_training_source_fraction=fraction,local_fit_not_generalization=True,visual_acceptance=False))


if __name__=='__main__':main()
