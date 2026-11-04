"""Teacher-affinity region interiors, with motion and sampling-phase KL.

The affinity masks are TRAINING ONLY. Inference remains exactly the validated
13-convolution INT8 graph and the fixed Y/Co/Cg categorical decoder.
"""
import argparse,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_equalized_regions import Training as Previous
from .train_luma_regions import ROOT,RUN,SLICES,labels,expected,probabilities,warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out):
    out.mkdir(exist_ok=False);base=ROOT/'runs'/RUN/'equalized-qat'
    (out/'data.json').write_bytes((base/'data.json').read_bytes());cfg=json.loads((base/'config.json').read_text())
    initial=base/'qat_checkpoint_001500.pt';cfg.update(seed=2026100431,qat_steps=1500,qat_lr=.00012,initial=str(initial),initial_sha256=file_hash(initial),
        region_offsets=[1,3,7],region_target_rgb255_threshold=4,region_kl_weight=4,temporal_kl_weight=2,phase_kl_weight=3)
    write_json(out/'config.json',cfg);binding=json.loads((base/'binding.json').read_text());binding['code'][str(Path(__file__))]=file_hash(__file__)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


def symmetric_kl(a,b):
    a=a.clamp_min(1e-7);b=b.clamp_min(1e-7)
    return .5*((a-b)*(a.log()-b.log())).sum(1)


class Training(Previous):
    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);p=self.net(x);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        lab=labels(t);pr=probabilities(p);ce=sum(F.cross_entropy(cut(p[:,s]),cut(l)) for s,l in zip(SLICES,lab))/3
        color=F.l1_loss(cut(expected(p)),cut(t));region=[];temporal=[];phase=[];coverage=[]
        with torch.no_grad():valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)])
        shifted=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];shiftpr=probabilities(self.net(shifted))
        for q,l,sq in zip(pr,lab,shiftpr):
            for d in self.cfg['region_offsets']:
                for axis in (-1,-2):
                    aa=[slice(None)]*4;bb=aa.copy();aa[axis]=slice(None,-d);bb[axis]=slice(d,None);aa=tuple(aa);bb=tuple(bb)
                    with torch.no_grad():
                        rgb_same=(t[aa]-t[bb]).abs().amax(1)<self.cfg['region_target_rgb255_threshold']/128
                        same=(l[aa[:1]+aa[2:]]==l[bb[:1]+bb[2:]]) & rgb_same
                    v=cut(symmetric_kl(q[aa],q[bb]));w=cut(same);region.append((v*w).sum()/w.sum().clamp_min(1));coverage.append(float(w.float().mean()))
            with torch.no_grad():same=(l[1:]==warp(l[:1,None].float(),flow)[:,0].round().long()).float()*valid[:,0]
            temporal.append((cut(symmetric_kl(q[1:],warp(q[:1],flow)))*cut(same)).sum()/cut(same).sum().clamp_min(1))
            phase.append(symmetric_kl(sq[...,m+sy:-m,m+sx:-m],q[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).mean())
        region=sum(region)/len(region);temporal=sum(temporal)/3;phase=sum(phase)/3
        ramp=min(1,(self.step+1)/150);loss=ce+.6*color+ramp*(4*region+2*temporal+3*phase)
        lr=self.cfg['qat_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['qat_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,mode=self.mode,loss=float(loss.detach()),ce=float(ce.detach()),color=float(color.detach()),region=float(region.detach()),temporal=float(temporal.detach()),phase=float(phase.detach()),
            affinity_fraction=sum(coverage)/len(coverage),grad=float(norm),native=native,lr=lr,sample=spec,seconds=time.perf_counter()-start)


def preflight(out):
    tr=Training(out);tr.update(True);tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected_state={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected_state[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample']
    write_json(out/'preflight.json',dict(status='PASS',full720_pair_forward_backward=True,batch2=True,checkpoint_restore_error=err));print('CLEAN_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--stop',type=int,default=1500);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100431);out=ROOT/'runs'/RUN/'clean-qat'
    if a.action=='prepare':prepare(out);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'qat_metrics.jsonl').exists();tr.evaluate()
    write_json(out/f'launch_{tr.step:06d}.json',dict(start=tr.step,resume=str(a.resume),binding=tr.binding))
    with (out/'qat_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['qat_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('CLEAN_QAT',tr.step,round(row['loss'],4),round(row['region'],5),round(row['phase'],5),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'qat_checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('CLEAN_SEGMENT_DONE',tr.step,flush=True)


if __name__=='__main__':main()
