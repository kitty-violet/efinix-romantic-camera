"""Align flatness/temporal targets with the discrete hardware color target."""
import argparse,copy,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_coarse_skip_regions import Training as Previous
from .train_detail_rgb import Training as DataTraining
from .coarse_skip_net import CoarseSkipNet
from .train_hard_regions import hard_soft
from .train_clean_regions import symmetric_kl
from .train_luma_regions import ROOT,RUN,SLICES,labels,probabilities,centers,reconstruct,warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out,initial):
    state=torch.load(initial,map_location='cpu',weights_only=True);assert state['scales'] is None and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    out.mkdir(exist_ok=False)
    for name in ['data.json','graph.json']:(out/name).write_bytes((initial.parent/name).read_bytes())
    cfg=copy.deepcopy(state['config']);cfg.update(seed=2026100444,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=1500,fp32_lr=.0003,
        region_objective='same discrete teacher labels define flat regions; discrete temporal teacher; stronger actual block uniformity',
        objective_weights=dict(ce=1,color=2,flat=64,region=4,temporal=3,phase=3))
    write_json(out/'config.json',cfg);binding=copy.deepcopy(state['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(Previous):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        state=torch.load(p,map_location='cpu',weights_only=True);self.net=CoarseSkipNet().cuda().eval();self.net.load_state_dict(state['model']);self.net.scales=None
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        aug=self.rng.random()<.5;trainx=(x+torch.randn_like(x)*self.cfg['texture_noise_rgb255']/128).clamp(-1,127/128) if aug else x
        p=self.net(trainx);lab=labels(t);pr=probabilities(p);y=hard_soft(p)
        with torch.no_grad():
            target=reconstruct(*[c[l] for c,l in zip(centers(t),lab)]);code=torch.stack(lab,1)
        ce=sum(F.cross_entropy(cut(p[:,s]),cut(l)) for s,l in zip(SLICES,lab))/3
        color=F.l1_loss(cut(y),cut(target));flat=[];region=[]
        for d in [1,3,7]:
            for axis in [-1,-2]:
                aa=[slice(None)]*4;bb=aa.copy();aa[axis]=slice(None,-d);bb[axis]=slice(d,None);aa=tuple(aa);bb=tuple(bb)
                with torch.no_grad():same=(code[aa]==code[bb]).all(1,keepdim=True)
                err=F.relu((y[aa]-y[bb]).abs().mean(1,keepdim=True)-1/128)
                flat.append((cut(err)*cut(same)).sum()/cut(same).sum().clamp_min(1))
                for q in pr:
                    kl=cut(symmetric_kl(q[aa],q[bb]));w=cut(same[:,0]);region.append((kl*w).sum()/w.sum().clamp_min(1))
        flat=sum(flat)/len(flat);region=sum(region)/len(region)
        with torch.no_grad():
            valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
            td=(target[1:]-warp(target[:1],flow)).abs().mean(1,keepdim=True)
        temporal=(cut(F.relu((y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True)-td-1/128))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]]
        q=hard_soft(self.net(shift));phase=(q[...,m+sy:-m,m+sx:-m]-y[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        losses=dict(ce=ce,color=color,flat=flat,region=region,temporal=temporal,phase=phase);ramp=min(1,(self.step+1)/150)
        loss=sum(v*self.cfg['objective_weights'][k]*(ramp if k in ('flat','region','temporal','phase') else 1) for k,v in losses.items())
        lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},grad=float(norm),lr=lr,native=native,noise=aug,sample=spec,seconds=time.perf_counter()-start)


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err));print('ALIGNED_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--initial',type=Path);p.add_argument('--stop',type=int,default=1500);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/'runs'/RUN/'aligned-regions-fp32'
    if a.action=='prepare':assert a.initial;prepare(out,a.initial);return
    configure_workspace();contract_guard();set_reproducible(2026100444)
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('ALIGNED_TRAIN',tr.step,round(row['loss'],4),round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('ALIGNED_DONE',tr.step,flush=True)


if __name__=='__main__':main()
