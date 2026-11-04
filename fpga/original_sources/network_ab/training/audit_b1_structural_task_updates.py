"""Compare actual parameter-space task directions and short saved-Adam steps.

Three fixed TRAIN crops; isolate appearance tasks only (no temporal/phase
claims). Every arm restores identical weights and optimizer moments. This
diagnostic is not a new deployable/generalized candidate.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_narrow_recall_native import objective
from .b1_local_rgb_envelope import local_envelope,core_error_variation
from .whitened_b1_optimizer import WhitenedB1Adam,transform
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def ordered_parameters(net):
    return [v for k,w in net.weight.items() for v in (w,net.bias[k])]


def restored_optimizer(net,saved):
    parameters=ordered_parameters(net);groups=[];stats={}
    assert len(parameters)==len(saved['param_groups'])
    for p,g in zip(parameters,saved['param_groups']):
        group={k:copy.deepcopy(v) for k,v in g.items() if k!='params'};group['params']=[p];groups.append(group)
        if 'node_key' in g:
            pp=g['input_whitener'];dim=pp.shape[-1]
            stats[g['node_key']]=dict(mean=g['patch_mean'],covariance=torch.eye(dim).expand_as(pp).clone())
    opt=WhitenedB1Adam(groups,net,stats);opt.load_state_dict(copy.deepcopy(saved));return opt


def transformed(grads,net,saved):
    result=[]
    for i,k in enumerate(net.weight):
        group=saved['param_groups'][2*i];gw,gb=grads[2*i:2*i+2]
        ga=transform(gw.flatten(1)-gb[:,None]*group['patch_mean'],group['input_whitener'].transpose(-1,-2)).reshape_as(gw)
        result.extend((ga,gb))
    return result


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=8);ap.add_argument('--include-projected',action='store_true')
    ap.add_argument('--guarded-only',action='store_true');ap.add_argument('--finite-check',action='store_true');a=ap.parse_args();torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);base.__class__=ReadoutB1
    optstate=state['optimizer'];byid={r['id']:r for r in m.load_data(a.data)['training']}
    mining=json.loads(Path(a.mining).read_text('utf-8'))['records'];xs=[];ts=[];ls=[];ids=[]
    for i in (1,5,6):
        spec=mining[i];row=byid[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        for key,dest in (('path',xs),('target',ts)):
            dest.append(torch.from_numpy(m.t.r.read(row[key])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float()/128-1)
        ls.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)));ids.append(row['id'])
    x=torch.stack(xs);t=torch.stack(ts)[...,40:-40,40:-40];lab=torch.stack(ls)[...,40:-40,40:-40]
    def terms(net,index=None):
        xx=x if index is None else x[index:index+1];tt=t if index is None else t[index:index+1];ll=lab if index is None else lab[index:index+1]
        p=net(xx)[0][...,40:-40,40:-40]
        main,_=objective(p,tt,ll);colour,_=local_envelope(p,tt,ll);core=core_error_variation(p,tt,ll)
        return [main,32*colour,16*core]
    gradients=[]
    for i,rid in enumerate(ids):
        costs=terms(base,i);params=ordered_parameters(base)
        gg=[torch.autograd.grad(v,params,retain_graph=True) for v in costs]
        axes={}
        for name,gl in [('effective_coefficients',gg),('fixed_whitened_coordinates',[transformed(g,base,optstate) for g in gg])]:
            flat=torch.stack([torch.cat([v.flatten() for v in g]) for g in gl]).double()
            gram=flat@flat.T;norm=gram.diag().sqrt();cos=gram/(norm[:,None]*norm[None]).clamp_min(1e-20)
            axes[name]=dict(norm=norm.tolist(),cosine=cos.tolist())
        gradients.append(dict(id=rid,costs=[float(v.detach()) for v in costs],axes=axes))
        print('TASK_PARAMETER_GRADIENT',gradients[-1],flush=True)
    arms={}
    names=['structure_only','structure_plus_guards']+(['projected_guards'] if a.include_projected else [])
    if a.guarded_only:names=['guarded_adam_step']
    for name in names:
        net=copy.deepcopy(base);opt=restored_optimizer(net,optstate);history=[]
        if name=='projected_guards':
            from .b1_projected_guard_optimizer import ProjectedGuardAdam,regression
            regression();opt.__class__=ProjectedGuardAdam
        if name=='guarded_adam_step':
            from .b1_guarded_adam_step import GuardedStepAdam,regression
            regression();opt.__class__=GuardedStepAdam
        for step in range(a.steps+1):
            costs=terms(net);history.append(dict(step=step,values=[float(v.detach()) for v in costs]))
            if step==a.steps:break
            opt.zero_grad();loss=costs[0] if name=='structure_only' else sum(costs)
            if name=='projected_guards':
                opt.guard_gradients=tuple(g.detach() for g in torch.autograd.grad(costs[1]+costs[2],ordered_parameters(net),retain_graph=True))
            if name=='guarded_adam_step':
                opt.task_gradients=[tuple(g.detach() for g in torch.autograd.grad(c,ordered_parameters(net),retain_graph=True)) for c in costs[1:]]
                if a.finite_check:opt.finite_check=(torch.stack(costs).detach(),lambda:torch.stack(terms(net)))
            loss.backward()
            opt.prepare_gradients();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
            if name in ('projected_guards','guarded_adam_step'):history[-1]['projection']=opt.last_projection
            if a.finite_check:history[-1]['finite_check']=opt.last_finite_check
        arms[name]=history;write_json(out/'partial.json',dict(gradients=gradients,arms=arms));print('TASK_UPDATE_ARM',name,history[0],history[-1],flush=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(gradients=gradients,arms=arms,checkpoint_sha256=file_hash(a.checkpoint),steps=a.steps,
        same_saved_optimizer=True,isolated_appearance_only=True,no_temporal_or_generalization_claim=True,not_a_weight_candidate=True))


if __name__=='__main__':main()
