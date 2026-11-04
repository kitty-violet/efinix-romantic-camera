"""Add an explicit final720 activation double-well to patch matching.

Extends the previous shape-only direction probe into actual training. Eight
fixed projection banks include RGB DC and spatial pattern directions, at
three scales; coarse content anchors keep the scene. There is no critic to
reward spurious invented texture. Inspired by GPDM, not a paper reproduction.
All distribution operations are training-only; actual inference is fixed B1.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_structural_task_updates import restored_optimizer
from .fit_b1_final720_code_distribution import Final720CodeB1 as ReadoutB1
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def banks(device):
    gen=torch.Generator().manual_seed(8267);out=[]
    for _ in range(8):
        w=torch.randn(64,3,7,7,generator=gen)
        # Half are shape-only, half retain colour; no per-channel sorting.
        w[:32]-=w[:32].mean((-1,-2),keepdim=True)
        w=w/w.flatten(1).norm(dim=1)[:,None,None,None];out.append(w.to(device))
    return out


def signature(z,w,scale):
    if scale>1:z=F.avg_pool2d(z,scale)
    a=F.conv2d(z,w);cell=max(8,32//scale)
    h=a.shape[-2]//cell*cell;ww=a.shape[-1]//cell*cell
    return a[...,:h,:ww].unfold(2,cell,cell).unfold(3,cell,cell).flatten(-2).sort(-1).values


def regression(device):
    w=banks(device)[0];yy,xx=torch.meshgrid(torch.arange(80,device=device),torch.arange(80,device=device),indexing='ij')
    t=torch.where(xx>38+.2*yy,-.5,.5)[None,None].expand(1,3,-1,-1).clone()
    ref=signature(t,w,1);assert (signature(t,w,1)-ref).abs().mean()==0
    blur=F.avg_pool2d(F.pad(t,(4,)*4,mode='reflect'),9,1).requires_grad_()
    loss=(signature(blur,w,1)-ref).abs().mean();grad=torch.autograd.grad(loss,blur)[0]
    assert loss>.01 and (grad*(t-blur.detach())).sum()<0 and torch.isfinite(grad).all()
    checker=torch.where((xx+yy)%2==0,-.5,.5)[None,None].expand_as(t)
    assert (signature(checker,w,1)-ref).abs().mean()>.05
    return dict(exact_target_zero=True,blur_gradient=True,checker_rejected=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=2400)
    ap.add_argument('--binary-weight',type=float,default=1.)
    a=ap.parse_args();configure_workspace();set_reproducible(8267);torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);checks=regression('cuda')
    state,net=load(a.checkpoint);net.__class__=ReadoutB1;net.cuda()
    rows={r['id']:r for r in m.load_data(a.data)['training']};rr=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[rr[i] for i in (1,5,6)]
    xs=[];ts=[]
    for s in specs:
        row=rows[s['row']['id']];xx,yy=s['xy'];w,h=s['wh']
        for key,dst in (('path',xs),('target',ts)):dst.append(m.t.r.read(row[key])[yy:yy+h,xx:xx+w])
    tensor=lambda ar:torch.from_numpy(np.stack(ar).transpose(0,3,1,2).copy()).float().cuda()/128-1
    x=tensor(xs);t=tensor(ts);cut=lambda z:z[...,40:-40,40:-40]
    with torch.no_grad():old=net(x)[0];bank=banks('cuda');target=[[signature(cut(t),w,s) for s in (1,2,4)] for w in bank]
    opt=restored_optimizer(net,state['optimizer']);opt.state.clear()
    def update(step):
        opt.zero_grad(set_to_none=True);nodes=net(x,all_nodes=True);p=cut(nodes['rgb_head']/128);i=step%len(bank)
        latent=cut(nodes['up2_high_dw'])/127
        binary=(latent*(1-latent)).mean()
        dist=sum((signature(p,bank[i],s)-target[i][j]).abs().mean() for j,s in enumerate((1,2,4)))/3
        coarse=sum((F.avg_pool2d(p,k)-F.avg_pool2d(cut(t),k)).abs().mean() for k in (8,16,32))/3
        loss=8*dist+2*coarse+a.binary_weight*binary;loss.backward();opt.prepare_gradients()
        gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);assert gn>0 and torch.isfinite(loss);opt.step()
        return float(loss.detach()),float(dist.detach()),float(coarse.detach()),float(binary.detach())
    initial=copy.deepcopy(net.state_dict());initialopt=copy.deepcopy(opt.state_dict())
    update(0);torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt')
    expected=update(1);snap=copy.deepcopy(net.state_dict());disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True)
    net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);observed=update(1)
    err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in snap.items());assert err==0 and expected==observed
    net.load_state_dict(initial);opt.load_state_dict(initialopt)
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),code=hashes,specs=specs,local_TRAIN_only=True,
        data_sha256=file_hash(a.data),checkpoint_sha256=file_hash(a.checkpoint),aligned_fine_pixel_loss=False,
        projection_banks=8,projections_per_bank=64,scales=[1,2,4],inference_additions=0))
    write_json(out/'preflight.json',dict(regression=checks,gpu_forward_backward_minibatch=True,disk_next_step_error=err))
    print('PATCH_DISTRIBUTION_PREFLIGHT_PASS realGPU disk0',flush=True)
    history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);started=time.monotonic()
    def review(step):
        with torch.no_grad():
            raw=net.export();assert len(raw)==20480;nodes=net(x,all_nodes=True)
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items());p=actual['rgb_head']/128
            dist=sum((signature(cut(p),bank[i],s)-target[i][j]).abs().mean() for i in range(8) for j,s in enumerate((1,2,4)))/24
        board=Image.new('RGB',(960,792),'#17202a');draw=ImageDraw.Draw(board)
        for i in range(3):
            for j,(title,z) in enumerate((('原片',x),('原目标',t),('初始B1',old),('实际B1·局部诊断',p))):
                board.paste(m.t.r.c.rgb_image(cut(z)[i]),(j*240,i*264+24));draw.text((j*240+3,i*264+3),title,font=font,fill='white')
        board.save(out/f'comparison_{step:06d}.png')
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,contract_sha256=state['contract_sha256'],
            residual_encoding=net.package['residual_encoding'],qat=True,local_fit_only=True,visual_acceptance=False,
            training_provenance=dict(args=vars(a),code=hashes)),out/f'candidate_{step:06d}.pt')
        (out/f'coefficients_{step:06d}.bin').write_bytes(raw)
        row=dict(step=step,mae=float(cut(p-t).abs().mean()*128),patch_distribution=float(dist),actual_all_nodes_error=0,latent_mid_fraction=float(((actual['up2_high_dw'][...,40:-40,40:-40]>12)&(actual['up2_high_dw'][...,40:-40,40:-40]<115)).float().mean()))
        history.append(row);write_json(out/'history.json',history);print('PATCH_DISTRIBUTION_REVIEW',row,flush=True)
    review(0)
    for step in range(a.steps):
        values=update(step)
        if (step+1)%100==0:print('PATCH_DISTRIBUTION_STEP',step+1,values,flush=True)
        if (step+1)%400==0 or step+1==a.steps:
            assert all(file_hash(k)==v for k,v in hashes.items());review(step+1)
            if (out/'STOP_REQUEST').exists():break
    write_json(out/'completion.json',dict(history=history,seconds=time.monotonic()-started,local_TRAIN_only=True,
        extra_inference_ops=0,visual_acceptance=False,video_not_validated=True))


if __name__=='__main__':main()
