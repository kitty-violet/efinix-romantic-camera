"""Project the proposed Adam step, not its input gradient, onto task descent.

Training only: the deployed B1 graph is unchanged. The local constraints do
not prove descent after finite quantization; callers must measure that.
"""
import itertools
import torch
from .whitened_b1_optimizer import WhitenedB1Adam, transform
from .train_b1_quantum_regions import RowScaledAdam


def constrain_step(proposal, gradients, metric, margin=.02):
    # Solve the <=3 constraint Euclidean QP by enumerating active sets.
    root=metric.double().clamp_min(1e-20).sqrt()
    z=proposal.double()/root
    a=gradients.double()*root[None]
    norms=a.norm(dim=1);keep=norms>1e-20
    a=a[keep]/norms[keep,None]
    rhs=torch.full((len(a),),-margin*float(z.norm()),device=z.device,dtype=z.dtype)
    def solve(b):
        best=None;score=float('inf')
        for n in range(len(a)+1):
            for ids in itertools.combinations(range(len(a)),n):
                if ids:
                    aa=a[list(ids)];q=aa@z-b[list(ids)]
                    gram=aa@aa.T
                    lam=torch.linalg.pinv(gram,rtol=1e-10,hermitian=True)@q
                    if (lam < -1e-9).any():continue
                    v=z-aa.T@lam
                else:v=z
                if (a@v-b>1e-8*(1+z.norm())).any():continue
                dist=float((v-z).square().sum())
                if dist<score:best,score=v,dist
        return best
    result=solve(rhs);relaxed=result is None
    if relaxed:result=solve(rhs*0)
    assert result is not None
    delta=(result*root).to(proposal.dtype)
    return delta,dict(proposed_task_dot=(gradients.double()@proposal.double()).tolist(),
        corrected_task_dot=(gradients.double()@delta.double()).tolist(),
        relative_correction=float((result-z).norm()/z.norm().clamp_min(1e-20)),
        margin_relaxed=relaxed)


class GuardedStepAdam(WhitenedB1Adam):
    def prepare_gradients(self):
        # Two stored guard gradients; primary includes every other loss.
        guards=[]
        for raw in self.task_gradients:
            transformed=[]
            for i,k in enumerate(self.net.weight):
                gw,gb=raw[2*i:2*i+2];group=self.param_groups[2*i]
                ga=transform(gw.flatten(1)-gb[:,None]*group['patch_mean'].to(gw.device),
                    group['input_whitener'].to(gw.device).transpose(-1,-2)).reshape_as(gw)
                transformed.extend((ga,gb))
            guards.append(torch.cat([g.flatten() for g in transformed]))
        super().prepare_gradients()
        total=torch.cat([g['params'][0].grad.flatten() for g in self.param_groups])
        self.step_tasks=torch.stack([total-sum(guards),*guards]).detach()
        self.task_gradients=None

    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None and self.prepared
        params=[g['params'][0] for g in self.param_groups]
        old=[p.clone() for p in params]
        RowScaledAdam.step(self)
        delta=torch.cat([(p-o).flatten() for p,o in zip(params,old)])
        metric=[]
        for group,p in zip(self.param_groups,params):
            state=self.state[p];beta2=group['betas'][1]
            denominator=(state['exp_avg_sq']/(1-beta2**float(state['step']))).sqrt()+group['eps']
            gain=group.get('row_gain',torch.ones_like(p)).to(p.device)
            metric.append((group['lr']*gain/denominator).expand_as(p).flatten())
        corrected,self.last_projection=constrain_step(delta,self.step_tasks,torch.cat(metric))
        offset=0
        for p,o in zip(params,old):
            p.copy_(o+corrected[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
        proposal=[]
        for i,k in enumerate(self.net.weight):
            group=self.param_groups[2*i];w=self.net.weight[k];previous=old[2*i]
            dw=transform((w-previous).flatten(1),group['input_whitener'].to(w.device)).reshape_as(w)
            dc=self.net.bias[k]-old[2*i+1]
            proposal.append((k,previous,old[2*i+1],dw,dc.clone(),group))
        def apply(alpha):
          for k,previous,old_bias,dw,dc,group in proposal:
            w=self.net.weight[k]
            s=(getattr(self.net,k+'_m')/131072).float().flatten()[:,None,None,None]
            w.copy_((previous+alpha*dw).maximum(-127*s).minimum(127*s))
            deployed=(self.net.round(w/s).clamp(-127,127)-self.net.round(previous/s).clamp(-127,127))*s
            self.net.bias[k].copy_(old_bias+alpha*dc-(deployed.flatten(1)*group['patch_mean'].to(w.device)).sum(1))
        apply(1.)
        if getattr(self,'finite_check',None) is not None:
            initial,evaluate=self.finite_check;records=[];accepted=0.
            for alpha in (1.,.5,.25,.125,.0625):
                apply(alpha);values=evaluate().detach()
                ok=bool(torch.isfinite(values).all() and (values<=initial+1e-7).all())
                records.append(dict(alpha=alpha,values=values.tolist(),accepted=ok))
                if ok:accepted=alpha;break
            if accepted==0:apply(0.)
            self.last_finite_check=dict(initial=initial.tolist(),trials=records,accepted=accepted,
                optimizer_moments_retained=True)
            self.finite_check=None
        self.prepared=False;self.step_tasks=None


def regression():
    gen=torch.Generator().manual_seed(229)
    for _ in range(100):
        g=torch.randn(3,29,generator=gen,dtype=torch.float64)
        p=torch.randn(29,generator=gen,dtype=torch.float64)
        metric=torch.exp(torch.randn(29,generator=gen,dtype=torch.float64)*2)
        d,s=constrain_step(p,g,metric)
        assert max(s['corrected_task_dot'])<1e-7
        good=-g.sum(0)
        d2,_=constrain_step(good,torch.stack([good.neg()]*3),metric,margin=0)
        assert torch.allclose(d2,good,atol=1e-10)
    d,s=constrain_step(torch.tensor([1.,2.]),torch.tensor([[1.,0.],[-1.,0.],[0.,1.]]),torch.ones(2))
    assert abs(float(d[0]))<1e-6 and s['margin_relaxed']
    print('GUARDED_ADAM_STEP_PASS common_descent/identity/opposite/variable_metric',flush=True)


if __name__=='__main__':regression()
