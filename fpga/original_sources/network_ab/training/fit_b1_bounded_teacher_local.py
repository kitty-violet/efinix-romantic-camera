"""Controlled3-TRAIN comparison: existing region target vs bounded reference.

Identical actual B1 initialization, optimizer coordinates, budget and loss.
This is only a local-fit diagnostic; never video/generalization acceptance.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_structural_task_updates import restored_optimizer
from .train_b1_trainable_luma_readout import ReadoutB1
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from . import train_b1_minimum_area as m


def objective(p,t):
    pixel=(p-t).abs().mean()
    edge=sum((torch.diff(p,dim=d)-torch.diff(t,dim=d)).abs().mean() for d in (-1,-2))/2
    return 16*(pixel+.5*edge),dict(mae=float(pixel.detach()*128),edge=float(edge.detach()*128))


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','data','mining','reference','out'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--steps',type=int,default=600);a=ap.parse_args()
    configure_workspace();set_reproducible(220);torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False)
    source_files=[Path(__file__),Path(__file__).with_name('whitened_b1_optimizer.py'),
        Path(__file__).with_name('train_b1_quantum_regions.py'),Path(__file__).with_name('train_b1_learned_colour.py')]
    hashes={str(p):file_hash(p) for p in source_files}
    for p in source_files:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),source_sha256=hashes,local_TRAIN_only=True))
    state,base=load(a.checkpoint);base.__class__=ReadoutB1;base.cuda()
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    specs=json.loads(Path(a.mining).read_text('utf-8'))['records'];xs=[];old=[];new=[];ids=[]
    for index in (1,5,6):
        spec=specs[index];row=rows[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        xs.append(m.t.r.read(row['path'])[yy:yy+h,xx:xx+w]);old.append(m.t.r.read(row['target'])[yy:yy+h,xx:xx+w])
        ref=np.array(Image.open(Path(a.reference)/(row['id']+'_3.png')).convert('RGB'))
        assert ref.shape==(h,w,3)
        new.append(ref);ids.append(row['id'])
    tensor=lambda ar:torch.from_numpy(np.stack(ar).transpose(0,3,1,2).copy()).float().cuda()/128-1
    x=tensor(xs);targets={'original_region':tensor(old),'bounded_source':tensor(new)}
    cut=lambda v:v[...,40:-40,40:-40]
    def opt_for(net):
        opt=restored_optimizer(net,state['optimizer']);opt.state.clear();return opt
    def step(net,opt,t):
        opt.zero_grad(set_to_none=True);loss,values=objective(cut(net(x)[0]),cut(t));loss.backward()
        opt.prepare_gradients();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        assert torch.isfinite(loss) and gn>0
        return values
    with torch.no_grad():before=base(x,all_nodes=True)
    actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(base.export()),device='cuda')(base.round(x*128),True)
    assert all(torch.equal(v,actual[k]) for k,v in before.items())
    del before,actual
    # Real disk resume after actual forward/backward updates, same next batch.
    test=copy.deepcopy(base);opt=opt_for(test);step(test,opt,targets['bounded_source'])
    torch.save(dict(model=test.state_dict(),optimizer=opt.state_dict()),out/'preflight.pt')
    expected=step(test,opt,targets['bounded_source']);snapshot=copy.deepcopy(test.state_dict())
    restored=torch.load(out/'preflight.pt',map_location='cpu',weights_only=True)
    test2=copy.deepcopy(base);test2.load_state_dict(restored['model']);opt2=opt_for(test2);opt2.load_state_dict(restored['optimizer'])
    observed=step(test2,opt2,targets['bounded_source']);err=max(float((v-snapshot[k]).abs().max()) for k,v in test2.state_dict().items())
    assert err==0 and expected==observed
    write_json(out/'preflight.json',dict(actual_all_nodes_error=0,disk_resume_next_step_error=err,
        real_GPU_forward_backward_minibatch=True,local_TRAIN_only=True));print('BOUNDED_TEACHER_PREFLIGHT',err,flush=True)
    del test,test2,opt,opt2,snapshot;torch.cuda.empty_cache()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);history={};start=time.monotonic()
    def review(net,name,iteration):
        with torch.no_grad():
            nodes=net(x,all_nodes=True);raw=net.export()
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            images=[np.array(m.t.r.c.rgb_image(p)) for p in nodes['rgb_head']/128]
        board=Image.new('RGB',(1280,1044),'#17202a');draw=ImageDraw.Draw(board)
        for i,rid in enumerate(ids):
            for j,(title,rgb) in enumerate((('原片',xs[i]),('原区域目标',old[i]),('局部参照·非B1',new[i]),('实际B1 '+name,images[i]))):
                # Only centre240 was supervised; outline it explicitly.
                board.paste(Image.fromarray(rgb),(320*j,348*i+28));draw.text((320*j+3,348*i+3),title,font=font,fill='white')
                draw.rectangle((320*j+40,348*i+68,320*j+279,348*i+307),outline='#d49b38',width=1)
            Image.fromarray(images[i]).save(out/f'{name}_{iteration:06d}_{rid}.png')
        board.save(out/f'{name}_{iteration:06d}_comparison.png')
        saved=copy.deepcopy(state);saved.update(model={k:v.cpu() for k,v in net.state_dict().items()},
            optimizer=current_opt.state_dict(),step=iteration,local_TRAIN_only=True,generalization_claim=False,
            bounded_reference=name=='bounded_source',visual_acceptance=False,
            args=dict(vars(a),training_class='BoundedTeacherLocalDiagnostic'),
            objective='16*(RGB_L1+.5*RGB_gradient_L1)',source_sha256=hashes)
        torch.save(saved,out/f'{name}_{iteration:06d}.pt')
    # Both arms are retained; do not compare raw MAE as equal target difficulty.
    for name,t in targets.items():
        net=copy.deepcopy(base);current_opt=opt_for(net);hh=[]
        with torch.no_grad():hh.append(dict(step=0,**objective(cut(net(x)[0]),cut(t))[1]))
        for it in range(1,a.steps+1):
            values=step(net,current_opt,t)
            if it%100==0:print('BOUNDED_TEACHER_STEP',name,it,values,flush=True)
            if it%200==0 or it==a.steps:
                assert all(file_hash(p)==v for p,v in hashes.items())
                with torch.no_grad():values=objective(cut(net(x)[0]),cut(t))[1]
                hh.append(dict(step=it,**values));review(net,name,it)
                history[name]=hh;write_json(out/'history.json',history)
        del net,current_opt;torch.cuda.empty_cache()
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'completion.json',dict(history=history,seconds=time.monotonic()-start,
        checkpoint_sha256=file_hash(a.checkpoint),actual_all_nodes_error=0,
        source_ids=ids,identical_initial_weights=True,identical_fresh_optimizer=True,
        no_generalization_or_video_claim=True,visual_acceptance=False))


if __name__=='__main__':main()
