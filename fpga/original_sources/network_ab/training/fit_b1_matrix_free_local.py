"""Damped matrix-free Gauss-Newton diagnostic of the plain contract model.

Fixed TRAIN batch/BN, raw input, Charbonnier pixel objective. IRLS weights are
held constant inside each CG solve. This is a local optimization diagnostic,
not a reproduction of Martens' full HF algorithm, a capacity bound, or a
deployable quantized result. https://icml.cc/2010/papers/458.pdf
"""
import argparse,copy,json,math,shutil,time
from pathlib import Path
import numpy as np,torch
from torch.func import functional_call,jvp,vjp
from PIL import Image,ImageDraw,ImageFont
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def dot(a,b):return (a.double()*b.double()).sum()


def cg(apply,b,steps,inverse_diagonal=None):
    z=torch.zeros_like(b);r=b.clone();q=r if inverse_diagonal is None else inverse_diagonal*r
    d=q.clone();rr=dot(r,q);initial=float(dot(r,r))
    for k in range(steps):
        ad=apply(d);den=dot(d,ad);assert den>0 and torch.isfinite(den)
        alpha=(rr/den).to(b.dtype);z=z+alpha*d;r=r-alpha*ad
        residual=dot(r,r)
        if float(residual)<initial*1e-8:return z,k+1,float(residual/initial)
        q=r if inverse_diagonal is None else inverse_diagonal*r;new=dot(r,q)
        d=q+(new/rr).to(b.dtype)*d;rr=new
    return z,steps,float(dot(r,r)/initial)


def regression():
    A=torch.tensor([[4.,1.],[1.,2.]],dtype=torch.float64);b=torch.tensor([3.,-1.],dtype=torch.float64)
    x,_,_=cg(lambda v:A@v,b,4);assert torch.allclose(A@x,b,atol=1e-12)
    diagonal=torch.tensor([1e-6,1.,1e6],dtype=torch.float64);b=torch.ones(3,dtype=torch.float64)
    x,iters,_=cg(lambda v:diagonal*v,b,4,1/diagonal);assert iters==1 and torch.allclose(x*diagonal,b,atol=1e-12)
    print('MATRIX_FREE_CG_LINEAR_PASS',flush=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=60);ap.add_argument('--cg',type=int,default=24)
    ap.add_argument('--precondition',action='store_true')
    a=ap.parse_args();configure_workspace();set_reproducible(2026100819);torch.set_num_threads(2);regression()
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    specs=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[specs[k] for k in (1,5,6)]
    xx=[];tt=[]
    for s in specs:
        r=byid[s['row']['id']];x,y=s['xy'];w,h=s['wh']
        for k,dst in (('path',xx),('target',tt)):
            dst.append(torch.from_numpy(m.t.r.read(r[k])[y:y+h,x:x+w].transpose(2,0,1).copy()).float()/128-1)
    x,t=torch.stack(xx).cuda(),torch.stack(tt).cuda();cut=lambda z:z[...,40:-40,40:-40]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);net=build_fp32_model().cuda().eval()
    assert state['local_fit_only'] and not state['quantization_tested'] and state['contract_sha256']==net.contract_sha256
    net.load_state_dict(state['model']);buffers={k:v.detach().clone() for k,v in net.named_buffers()}
    keys=list(dict(net.named_parameters()));shapes=[p.shape for p in net.parameters()];sizes=[p.numel() for p in net.parameters()]
    def flatten():return torch.cat([p.detach().flatten() for p in net.parameters()])
    def unpack(v):return {k:q.reshape(s) for k,q,s in zip(keys,v.split(sizes),shapes)}
    def forward(v):return cut(functional_call(net,(unpack(v),buffers),(x,)))
    def objective(p):return ((p-cut(t)).square()+(1/128)**2).sqrt().mean()
    out=Path(a.out);out.mkdir(exist_ok=False);initial=flatten();initial_damping=.1 if a.precondition else .001;damping=initial_damping
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,code=hashes,checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),
        local_fit_only=True,quantization_tested=False,raw_source_input=True,objective='mean sqrt(error**2+(1/128)**2)',
        method='IRLS damped Gauss-Newton; fixed current Jacobian/weights in each CG',precision='FP32 no AMP/TF32'))
    def update(theta,lam):
        # All parameters, buffers and image data are fixed throughout the solve.
        p,pullback=vjp(forward,theta);residual=(p-cut(t)).detach();weight=(residual.square()+(1/128)**2).rsqrt();n=p.numel()
        gradient=pullback(weight*residual/n)[0].detach();old=float(objective(p).detach())
        diagonal=torch.ones_like(theta)
        if a.precondition:
            # Rademacher estimate of diag(J^T W J/N). Deterministic probes and
            # all other solve inputs are held fixed within each linear solve.
            generator=torch.Generator(device=theta.device).manual_seed(2026100820)
            diagonal.zero_()
            for _ in range(16):
                z=torch.randint(0,2,p.shape,device=p.device,generator=generator).float()*2-1
                g=pullback(z*(weight/n).sqrt())[0].detach();diagonal+=g.square()/16
            diagonal=diagonal.clamp_min(1e-8)
        def apply(v):
            tangent=jvp(forward,(theta,),(v,))[1].detach()
            return pullback(tangent*weight/n)[0].detach()+lam*diagonal*v
        delta,iters,rel=cg(apply,-gradient,a.cg,1/((1+lam)*diagonal) if a.precondition else None);assert dot(gradient,delta)<0
        ad=apply(delta)-lam*diagonal*delta
        with torch.no_grad():
            best=theta;new=old;accepted=0.;rho=0.
            for alpha in (1.,.5,.25,.125,.0625,.03125):
                trial=theta+alpha*delta;value=float(objective(forward(trial)))
                if value<new:
                    best=trial;new=value;accepted=alpha
                    expected=-alpha*float(dot(gradient,delta))-.5*alpha**2*float(dot(delta,ad))
                    rho=(old-new)/max(expected,1e-12);break
        if rho<.25:lam*=1.5
        elif rho>.75:lam*=2/3
        lam=min(1.,max(lam,1e-8))
        return best.detach(),lam,dict(old=old,new=new,alpha=accepted,rho=rho,damping=lam,cg_iterations=iters,cg_relative_residual=rel,
            gradient_norm=float(gradient.norm()),delta_norm=float(delta.norm()),diagonal_min=float(diagonal.min()),diagonal_max=float(diagonal.max()))
    # Independent adjoint identity checks the exact JVP/VJP pairing used in CG.
    theta=initial.clone();p,pullback=vjp(forward,theta);v=torch.randn_like(theta);v=v/v.norm();u=torch.randn_like(p);u=u/u.norm()
    tangent=jvp(forward,(theta,),(v,))[1];left=dot(tangent,u);right=dot(v,pullback(u)[0]);adjoint=float((left-right).abs())
    assert adjoint<1e-5,(float(left),float(right),adjoint)
    theta,damping,_=update(theta,damping)
    torch.save(dict(theta=theta,damping=damping),out/'restore_probe.pt')
    expected,el,er=update(theta,damping);disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True)
    actual,al,ar=update(disk['theta'],disk['damping']);error=float((actual-expected).abs().max())
    assert error==0 and el==al and er==ar
    write_json(out/'preflight.json',dict(status='PASS',gpu_forward_backward=True,adjoint_error=adjoint,disk_next_step_error=error))
    print('MATRIX_FREE_GPU_PREFLIGHT_PASS adjoint/disk0',flush=True)
    del pullback,p,tangent,u,v,expected,actual;theta=initial.clone();damping=initial_damping
    with torch.no_grad():baseline=forward(theta)
    history=[];last={};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);started=time.monotonic()
    for step in range(a.steps+1):
        if step%10==0 or step==a.steps:
            assert all(file_hash(p)==h for p,h in hashes.items())
            with torch.no_grad():p=forward(theta)
            rec=dict(step=step,mae=float((p-cut(t)).abs().mean()*128),objective=float(objective(p)),**last)
            history.append(rec);write_json(out/'history.json',history)
            board=Image.new('RGB',(720,792),'#17202a');draw=ImageDraw.Draw(board)
            for i in range(3):
                for j,(title,z) in enumerate((('固定目标',cut(t)),('初始FP32',baseline),('矩阵自由二阶·局部FP32',p))):
                    board.paste(m.t.r.c.rgb_image(z[i]),(240*j,264*i+24));draw.text((240*j+2,264*i+2),title,font=font,fill='white')
            board.save(out/f'comparison_{step:06d}.png')
            sd=copy.deepcopy(net.state_dict());sd.update({k:v.detach().clone() for k,v in unpack(theta).items()})
            torch.save(dict(model=sd,theta=theta,damping=damping,step=step,contract_sha256=net.contract_sha256,local_fit_only=True,
                quantization_tested=False,visual_acceptance=False,training_provenance=dict(args=vars(a),code=hashes)),out/f'float_{step:06d}.pt')
            print('MATRIX_FREE_LOCAL',rec,flush=True)
        if step==a.steps:break
        theta,damping,last=update(theta,damping)
        assert all(torch.equal(v,buffers[k]) for k,v in net.named_buffers())
    write_json(out/'results.json',dict(history=history,seconds=time.monotonic()-started,visual_acceptance=False,quantization_tested=False))


if __name__=='__main__':main()
