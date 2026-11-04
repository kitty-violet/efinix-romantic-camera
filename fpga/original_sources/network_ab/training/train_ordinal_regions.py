"""Ordered Y/Co/Cg training; export ordinary C32 convolution coefficients."""
import argparse,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .ordinal_coarse_net import OrdinalCoarseNet
from .train_aligned_regions import Training as Aligned
from .train_detail_rgb import Training as DataTraining
from .train_hard_regions import hard_soft
from .train_clean_regions import symmetric_kl
from .train_luma_regions import ROOT,RUN,SLICES,labels,probabilities,centers,reconstruct,warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out):
    assert not (out/'preflight.pt').exists() and not (out/'fp32_metrics.jsonl').exists()
    cfg=json.loads((out/'config.json').read_text());cfg['objective_weights']['surface']=8
    cfg['region_objective']='Ordered head, continuous color/within-region gradients and actual discrete flat/phase/temporal errors'
    write_json(out/'config.json',cfg)
    binding=json.loads((out/'binding.json').read_text());binding['code'][str(Path(__file__))]=file_hash(__file__)
    binding['config_sha256']=file_hash(out/'config.json');write_json(out/'binding.json',binding)


class Training(Aligned):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        state=torch.load(p,map_location='cpu',weights_only=True);self.net=OrdinalCoarseNet().cuda().eval();self.net.load_state_dict(state['model']);self.net.scales=None
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        aug=self.rng.random()<.5;trainx=(x+torch.randn_like(x)*self.cfg['texture_noise_rgb255']/128).clamp(-1,127/128) if aug else x
        nodes=self.net(trainx,True);p=nodes['rgb'];z=self.net.component_prediction(nodes['finish.dw']);smooth=reconstruct(*z.unbind(1))
        lab=labels(t);pr=probabilities(p);y=hard_soft(p)
        with torch.no_grad():target=reconstruct(*[c[l] for c,l in zip(centers(t),lab)]);code=torch.stack(lab,1)
        ce=sum(F.cross_entropy(cut(p[:,s]),cut(l)) for s,l in zip(SLICES,lab))/3
        color=F.l1_loss(cut(smooth),cut(t));flat=[];region=[];surface=[]
        for d in [1,3,7]:
            for axis in [-1,-2]:
                aa=[slice(None)]*4;bb=aa.copy();aa[axis]=slice(None,-d);bb[axis]=slice(d,None);aa=tuple(aa);bb=tuple(bb)
                with torch.no_grad():same=(code[aa]==code[bb]).all(1,keepdim=True)
                err=F.relu((y[aa]-y[bb]).abs().mean(1,keepdim=True)-1/128)
                flat.append((cut(err)*cut(same)).sum()/cut(same).sum().clamp_min(1))
                se=F.relu((smooth[aa]-smooth[bb]).abs().mean(1,keepdim=True)-(t[aa]-t[bb]).abs().mean(1,keepdim=True)-.5/128)
                surface.append((cut(se)*cut(same)).sum()/cut(same).sum().clamp_min(1))
                for q in pr:
                    kl=cut(symmetric_kl(q[aa],q[bb]));w=cut(same[:,0]);region.append((kl*w).sum()/w.sum().clamp_min(1))
        flat=sum(flat)/len(flat);region=sum(region)/len(region);surface=sum(surface)/len(surface)
        with torch.no_grad():
            valid=mask*((x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)<8/128)
            td=(target[1:]-warp(target[:1],flow)).abs().mean(1,keepdim=True)
        temporal=(cut(F.relu((y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True)-td-1/128))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]]
        q=hard_soft(self.net(shift));phase=(q[...,m+sy:-m,m+sx:-m]-y[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        losses=dict(ce=ce,color=color,flat=flat,region=region,surface=surface,temporal=temporal,phase=phase);ramp=min(1,(self.step+1)/150)
        loss=sum(v*self.cfg['objective_weights'][k]*(ramp if k not in ('ce','color') else 1) for k,v in losses.items())
        lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},grad=float(norm),lr=lr,native=native,noise=aug,sample=spec,seconds=time.perf_counter()-start)

    def save(self):
        path=self.out/f'fp32_checkpoint_{self.step:06d}.pt';torch.save(self.state(),path)
        # Deployment checkpoint deliberately has no optimizer/resume claim.
        ordinary=self.net.folded();torch.save(dict(model=ordinary.state_dict(),scales=None,mode='fp32',step=self.step,
            binding=self.binding,config=self.cfg,source=str(path),source_sha256=file_hash(path),training_only_factorization_removed=True),self.out/f'folded_{self.step:06d}.pt')


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    with torch.no_grad():
        x=torch.rand(1,3,720,1280,device='cuda')*2-1;before=tr.net(x);folded=tr.net.folded();after=folded(x);fold_err=float((before-after).abs().max())
        assert fold_err==0 and torch.equal(before,tr.net(x)), 'Export must leave the live model intact'
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,full720_fold_max_error=fold_err));print('ORDINAL_PREFLIGHT_PASS',err,fold_err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/'runs'/RUN/'ordinal-regions-fp32'
    if a.action=='prepare':prepare(out);return
    configure_workspace();contract_guard();set_reproducible(2026100446)
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('ORDINAL_TRAIN',tr.step,round(row['loss'],4),round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:tr.save();tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():tr.save();return
    print('ORDINAL_DONE',tr.step,flush=True)


if __name__=='__main__':main()
