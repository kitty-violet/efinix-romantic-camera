"""Train the finite-arena checked 41-pixel context model in real FP32."""
import argparse,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .context_region_net import ContextNet,plan
from .train_detail_rgb import Training as DataTraining
from .train_luma_regions import Training as LumaTraining,labels,probabilities,expected,SLICES,warp,ROOT,RUN
from .train_clean_regions import symmetric_kl
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out,initial):
    out.mkdir(exist_ok=False);base=ROOT/'runs'/RUN/'clean-qat'
    audit=ROOT/'runs'/RUN/'context-budget-v3/audit.json';budget=json.loads(audit.read_text())
    assert budget['source_sha256']==file_hash(Path(__file__).with_name('context_region_net.py'))
    (out/'data.json').write_bytes((base/'data.json').read_bytes())
    cfg=json.loads((base/'config.json').read_text());cfg.update(seed=2026100432,fp32_steps=2000,fp32_lr=.0004,margin=32,
        initial=str(initial),initial_sha256=file_hash(initial),budget_audit_sha256=file_hash(audit),precision='FP32 no AMP/TF32/EMA',
        architecture='two learned downsample stages, native stem boundary skip, fixed categorical output',region_offsets=[1,3,7,15])
    write_json(out/'config.json',cfg);write_json(out/'graph.json',plan())
    binding=json.loads((base/'binding.json').read_text());binding['code'].update({str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('context_region_net.py'),Path(__file__).with_name('train_clean_regions.py')]})
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(DataTraining):
    evaluate=LumaTraining.evaluate
    def __init__(self,out):
        super().__init__(out,'fp32');self.net=ContextNet().cuda().eval();p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        s=torch.load(p,map_location='cpu',weights_only=True);missing,extra=self.net.load_state_dict(s['model'],strict=False)
        assert not extra and missing and all(k.startswith('layers.context__') for k in missing)
        self.net.scales=None;self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    def update(self,native=None):
        assert self.net.scales is None
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);p=self.net(x);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        lab=labels(t);pr=probabilities(p);ce=sum(F.cross_entropy(cut(p[:,s]),cut(l)) for s,l in zip(SLICES,lab))/3
        color=F.l1_loss(cut(expected(p)),cut(t));region=[];temporal=[];phase=[]
        with torch.no_grad():valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)])
        shifted=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];shiftpr=probabilities(self.net(shifted))
        for q,l,sq in zip(pr,lab,shiftpr):
            for d in self.cfg['region_offsets']:
                for axis in (-1,-2):
                    aa=[slice(None)]*4;bb=aa.copy();aa[axis]=slice(None,-d);bb[axis]=slice(d,None);aa=tuple(aa);bb=tuple(bb)
                    with torch.no_grad():same=(l[aa[:1]+aa[2:]]==l[bb[:1]+bb[2:]]) & ((t[aa]-t[bb]).abs().amax(1)<4/128)
                    v=cut(symmetric_kl(q[aa],q[bb]));w=cut(same);region.append((v*w).sum()/w.sum().clamp_min(1))
            with torch.no_grad():same=(l[1:]==warp(l[:1,None].float(),flow)[:,0].round().long()).float()*valid[:,0]
            temporal.append((cut(symmetric_kl(q[1:],warp(q[:1],flow)))*cut(same)).sum()/cut(same).sum().clamp_min(1))
            phase.append(symmetric_kl(sq[...,m+sy:-m,m+sx:-m],q[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).mean())
        region=sum(region)/len(region);temporal=sum(temporal)/3;phase=sum(phase)/3;ramp=min(1,(self.step+1)/200)
        loss=ce+.6*color+ramp*(4*region+2*temporal+3*phase)
        lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,mode=self.mode,loss=float(loss.detach()),ce=float(ce.detach()),color=float(color.detach()),region=float(region.detach()),temporal=float(temporal.detach()),phase=float(phase.detach()),grad=float(norm),native=native,lr=lr,sample=spec,seconds=time.perf_counter()-start)


def preflight(out):
    tr=Training(out);tr.update(True);tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected_state={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected_state[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample']
    write_json(out/'preflight.json',dict(status='PASS',actual_native720_pair_forward_backward=True,batch2=True,independent_restore_error=err,fp32=True));print('CONTEXT_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--initial',type=Path);p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100432);out=ROOT/'runs'/RUN/'context-fp32'
    if a.action=='prepare':assert a.initial;prepare(out,a.initial);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists();tr.evaluate()
    write_json(out/f'launch_{tr.step:06d}.json',dict(start=tr.step,resume=str(a.resume),binding=tr.binding))
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('CONTEXT_FP32',tr.step,round(row['loss'],4),round(row['region'],5),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('CONTEXT_SEGMENT_DONE',tr.step,flush=True)


if __name__=='__main__':main()
