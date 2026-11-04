"""Local actual-B1 test without fine aligned-pixel/contour regression.

Coarse colour/content anchors plus an unconditional centred-RGB patch critic.
Previous conditional-band critics retained strong aligned fine objectives and
limited the critic relative to them. Here there is no fine regression or
target-gradient projection. Not a reproduction of CartoonGAN/White-box GAN.
No discriminator, blur, or centring survives export to the B1 package.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_structural_task_updates import restored_optimizer
from .train_b1_trainable_luma_readout import ReadoutB1
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


class PatchJudge(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList()
        ci=3
        for co,stride in ((24,1),(32,2),(48,1),(1,1)):
            self.layers.append(nn.utils.spectral_norm(nn.Conv2d(ci,co,3,stride,0)));ci=co
    def forward(self,x):
        x=4*(x-F.avg_pool2d(F.pad(x,(7,)*4,mode='reflect'),15,1))
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i+1<len(self.layers):x=F.leaky_relu(x,.2)
        return x


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--adv',type=float,default=.25)
    a=ap.parse_args();configure_workspace();set_reproducible(8235);torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);state,net=load(a.checkpoint);net.__class__=ReadoutB1;net.cuda()
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    records=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[records[i] for i in (1,5,6)];xs=[];ts=[]
    for s in specs:
        row=byid[s['row']['id']];xx,yy=s['xy'];w,h=s['wh']
        for key,dst in (('path',xs),('target',ts)):dst.append(m.t.r.read(row[key])[yy:yy+h,xx:xx+w])
    tensor=lambda ar:torch.from_numpy(np.stack(ar).transpose(0,3,1,2).copy()).float().cuda()/128-1
    x=tensor(xs);t=tensor(ts);cut=lambda v:v[...,40:-40,40:-40]
    with torch.no_grad():old=net(x)[0].detach()
    d=PatchJudge().cuda();do=torch.optim.Adam(d.parameters(),lr=.00015,betas=(.5,.99))
    opt=restored_optimizer(net,state['optimizer']);opt.state.clear()
    blur=F.avg_pool2d(F.pad(cut(t),(3,)*4,mode='reflect'),7,1)
    def dstep(p,step):
        d.train().requires_grad_(True);do.zero_grad(set_to_none=True)
        # Cycle positions, every patch keeps its real neighbours.
        yy=(step*17)%80;xx=(step*23)%80;crop=lambda z:z[...,yy:yy+160,xx:xx+160]
        real=crop(cut(t)).detach();neg=[crop(p).detach(),crop(blur),crop(cut(x))]
        r=d(real);f=d(torch.cat(neg));loss=.5*(r-1).square().mean()+.5*f.square().mean()
        loss.backward();do.step();return float(loss.detach())
    for it in range(100):dstep(cut(old),it)
    d.eval().requires_grad_(False)
    def update(step):
        opt.zero_grad(set_to_none=True);p=net(x)[0];pc=cut(p);tc=cut(t)
        dl=dstep(pc,step);d.eval().requires_grad_(False)
        coarse=sum((F.avg_pool2d(pc,k)-F.avg_pool2d(tc,k)).abs().mean() for k in (8,16,32))/3
        adv=(d(pc)-1).square().mean();loss=4*coarse+a.adv*adv
        # No aligned fine RGB/gradient/region loss and no gradient projection.
        loss.backward();opt.prepare_gradients();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
        assert gn>0 and torch.isfinite(loss);opt.step()
        return float(loss.detach()),float(coarse.detach()),float(adv.detach()),dl
    def snap():return dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),judge=copy.deepcopy(d.state_dict()),judge_optimizer=copy.deepcopy(do.state_dict()))
    def restore(s):net.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);d.load_state_dict(s['judge']);do.load_state_dict(s['judge_optimizer'])
    initial=snap();update(0);torch.save(snap(),out/'restore_probe.pt');expected=update(1);after=snap()
    restore(torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True));observed=update(1)
    err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in after['model'].items())
    de=max(float((v-d.state_dict()[k]).abs().max()) for k,v in after['judge'].items())
    assert err==0 and de==0 and expected==observed;restore(initial)
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,code=hashes,checkpoint_sha256=file_hash(a.checkpoint),
        data_sha256=file_hash(a.data),local_TRAIN_only=True,fine_aligned_pixel_loss=False,critic_in_inference=False))
    write_json(out/'preflight.json',dict(real_gpu_forward_backward=True,disk_next_step_model_error=err,judge_error=de))
    print('UNALIGNED_CRITIC_PREFLIGHT_PASS model0 judge0 disk_next0',flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);history=[];start=time.monotonic()
    def review(step):
        with torch.no_grad():
            raw=net.export();nodes=net(x,all_nodes=True)
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items());p=actual['rgb_head']/128
            dl=dict(real=float(d(cut(t)).mean()),fake=float(d(cut(p)).mean()),blur=float(d(blur).mean()),photo=float(d(cut(x)).mean()))
        board=Image.new('RGB',(960,792),'#17202a');draw=ImageDraw.Draw(board)
        for i in range(3):
            for j,(title,z) in enumerate((('原片',x),('固定目标',t),('初始实际B1',old),('实际B1·局部诊断',p))):
                board.paste(m.t.r.c.rgb_image(cut(z)[i]),(240*j,i*264+24));draw.text((j*240+3,i*264+3),title,font=font,fill='white')
        board.save(out/f'comparison_{step:06d}.png')
        saved=snap();saved.update(step=step,residual_encoding=copy.deepcopy(net.package['residual_encoding']),contract_sha256=state['contract_sha256'],
            local_fit_only=True,qat=True,visual_acceptance=False,training_provenance=dict(args=vars(a),code=hashes))
        torch.save(saved,out/f'candidate_{step:06d}.pt');(out/f'coefficients_{step:06d}.bin').write_bytes(raw)
        row=dict(step=step,mae=float(cut(p-t).abs().mean()*128),critic=dl,integer_all_nodes_error=0)
        history.append(row);write_json(out/'history.json',history);print('UNALIGNED_CRITIC_REVIEW',row,flush=True)
    review(0)
    for step in range(a.steps):
        values=update(step)
        if (step+1)%100==0:print('UNALIGNED_CRITIC_STEP',step+1,values,flush=True)
        if (step+1)%400==0 or step+1==a.steps:
            assert all(file_hash(k)==v for k,v in hashes.items());review(step+1)
            if (out/'STOP_REQUEST').exists():break
    write_json(out/'completion.json',dict(seconds=time.monotonic()-start,history=history,extra_inference_ops=0,
        local_TRAIN_only=True,visual_acceptance=False,temporal_not_tested=True))


if __name__=='__main__':main()
