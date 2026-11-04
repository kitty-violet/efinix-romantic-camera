"""All-weight local B1 split-variable diagnostic with refreshed sharp targets.

Unlike the old fixed-prefix/self-contour test, all existing weights are free;
targets are recomputed from the actual current source prediction each round.
An output residual dual discourages repeatedly accepting the same soft fit.
This nonconvex alternating experiment has no convergence/quality guarantee.
Projection and dual exist only in training, never in the exported graph.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_structural_task_updates import restored_optimizer
from .train_b1_trainable_luma_readout import ReadoutB1
from .b1_output_projection import project,regression
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--rounds',type=int,default=4);ap.add_argument('--inner',type=int,default=300)
    ap.add_argument('--strength',type=float,default=.0004)
    a=ap.parse_args();configure_workspace();set_reproducible(8123);torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);checks=regression('cuda')
    state,net=load(a.checkpoint);net.__class__=ReadoutB1;net.cuda()
    assert all(p.requires_grad for p in net.parameters())
    data=m.load_data(a.data);rows={r['id']:r for r in data['training']}
    records=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[records[i] for i in (1,5,6)]
    xs=[];ts=[]
    for s in specs:
        row=rows[s['row']['id']];xx,yy=s['xy'];w,h=s['wh']
        for key,dst in (('path',xs),('target',ts)):
            dst.append(m.t.r.read(row[key])[yy:yy+h,xx:xx+w])
    tensor=lambda ar:torch.from_numpy(np.stack(ar).transpose(0,3,1,2).copy()).float().cuda()/128-1
    x=tensor(xs);t=tensor(ts);cut=lambda v:v[...,40:-40,40:-40]
    with torch.no_grad():old=net(x)[0].detach();q=project(old,a.strength);u=torch.zeros_like(q)
    opt=restored_optimizer(net,state['optimizer']);opt.state.clear()
    initial=copy.deepcopy(net.state_dict());initialopt=copy.deepcopy(opt.state_dict())
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__),Path(__file__).with_name('b1_output_projection.py')]))
    hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,code=hashes,
        data_sha256=file_hash(a.data),checkpoint_sha256=file_hash(a.checkpoint),local_TRAIN_only=True,
        all_existing_parameters_trainable=True,projection_in_inference=False,
        nonlinear_split_iteration_not_convergence_guarantee=True))
    def update():
        opt.zero_grad(set_to_none=True);p=net(x)[0]
        fit=(cut(p-q+u)).square().mean()
        colour=(F.avg_pool2d(cut(p),16)-F.avg_pool2d(cut(t),16)).square().mean()
        fidelity=(cut(p-old)).square().mean()
        loss=128*(fit+.15*colour+.05*fidelity)
        loss.backward();opt.prepare_gradients();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
        assert gn>0 and torch.isfinite(loss);opt.step()
        return float(loss.detach()),float(gn)
    update();torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),q=q,u=u),out/'restore_probe.pt')
    expected=update();snap=copy.deepcopy(net.state_dict())
    disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True)
    net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);q=disk['q'];u=disk['u'];observed=update()
    err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in snap.items());assert err==0 and expected==observed
    net.load_state_dict(initial);opt.load_state_dict(initialopt)
    write_json(out/'preflight.json',dict(regression=checks,gpu_forward_backward_minibatch=True,disk_next_step_error=err))
    print('ADAPTIVE_PROJECTION_PREFLIGHT_PASS disk0 all_parameters',flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);history=[];start=time.monotonic()
    def review(iteration):
        with torch.no_grad():
            raw=net.export();assert len(raw)==20480
            nodes=net(x,all_nodes=True);actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items());p=actual['rgb_head']/128
            row=dict(step=iteration,projection_mae=float(cut(p-q).abs().mean()*128),teacher_mae=float(cut(p-t).abs().mean()*128),
                dual_max=float(u.abs().max()*128),gradient_mean=float((torch.diff(cut(p),dim=-1).abs().mean()+torch.diff(cut(p),dim=-2).abs().mean())*64),
                all_integer_nodes_error=0)
        board=Image.new('RGB',(1200,792),'#17202a');draw=ImageDraw.Draw(board)
        for i in range(3):
            for j,(title,z) in enumerate((('原片',x),('原风格目标',t),('初始B1',old),('本轮训练投影·非推理',q),('实际B1',p))):
                board.paste(m.t.r.c.rgb_image(cut(z)[i]),(240*j,264*i+24));draw.text((240*j+3,264*i+3),title,font=font,fill='white')
        board.save(out/f'comparison_{iteration:06d}.png')
        saved=dict(model=net.state_dict(),optimizer=opt.state_dict(),step=iteration,residual_encoding=net.package['residual_encoding'],
            contract_sha256=state['contract_sha256'],q=q,u=u,local_fit_only=True,qat=True,visual_acceptance=False,
            training_provenance=dict(kind='adaptive output projection all-weight local diagnostic',args=vars(a),code=hashes))
        torch.save(saved,out/f'candidate_{iteration:06d}.pt');(out/f'coefficients_{iteration:06d}.bin').write_bytes(raw)
        history.append(row);write_json(out/'history.json',history);print('ADAPTIVE_PROJECTION_REVIEW',row,flush=True)
    review(0)
    for outer in range(a.rounds):
        if outer:
            with torch.no_grad():
                p=net(x)[0];u=u+(p-q);q=project(p+u,a.strength)
        for inner in range(a.inner):
            values=update()
            if (inner+1)%100==0:print('ADAPTIVE_PROJECTION_STEP',outer+1,inner+1,values,flush=True)
        assert all(file_hash(k)==v for k,v in hashes.items())
        review((outer+1)*a.inner)
        if (out/'STOP_REQUEST').exists():break
    write_json(out/'completion.json',dict(seconds=time.monotonic()-start,history=history,visual_acceptance=False,
        local_TRAIN_only=True,extra_inference_ops=0,temporal_video_not_validated=True))


if __name__=='__main__':main()
