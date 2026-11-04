"""Regress deployed region centers, not the teacher's continuous RGB texture.

Same C3 inference graph and palette. All added operations are training losses.
No validation frame enters training. Frozen earlier experiments remain intact.
"""
import argparse,copy,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_component_regions import Training as Prior,straight_colors
from .train_detail_rgb import Training as DataTraining
from .component_region_net import ComponentNet,plan
from .train_luma_regions import ROOT,RUN,centers,labels,warp,reconstruct
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def target_components(t):
    return torch.stack([v[l] for v,l in zip(centers(t),labels(t))],1)


def spatial(p,t,margin):
    cut=lambda z:z[...,margin:-margin,margin:-margin]
    # This target is exactly representable by the deployed scalar decoder.
    c=target_components(t);terms=[];flat=[]
    for d in (1,3,7):
        for axis in (-1,-2):
            a=[slice(None)]*4;b=a.copy();a[axis]=slice(None,-d);b[axis]=slice(d,None);a=tuple(a);b=tuple(b)
            pd=p[a]-p[b];td=c[a]-c[b];same=(td==0).all(1,keepdim=True)
            terms.append((cut(pd)-cut(td)).abs().mean())
            flat.append((cut(pd.abs())*cut(same)).sum()/(3*cut(same).sum()).clamp_min(1))
    return dict(center=F.l1_loss(cut(p),cut(c)),boundary=sum(terms)/len(terms),flat=sum(flat)/len(flat))


def prepare(out,initial,steps):
    s=torch.load(initial,map_location='cpu',weights_only=True)
    assert s['scales'] is None and all(file_hash(k)==v for k,v in s['binding']['code'].items())
    out.mkdir(exist_ok=False)
    for k in ('data.json','graph.json'):(out/k).write_bytes((initial.parent/k).read_bytes())
    cfg=copy.deepcopy(s['config']);cfg.update(seed=2026100450,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=steps,fp32_lr=.0004,
        objective_weights=dict(center=12,boundary=4,flat=16,temporal=4,phase=4),curriculum_probability=0,
        region_objective='Exact deployed component centers; zero interior gradient; matched signed boundaries',
        temporal_reference='Original source residual capped3+1RGB; no teacher-delta permission')
    write_json(out/'config.json',cfg);binding=copy.deepcopy(s['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    # A perfect region map must incur zero spatial loss, including boundaries.
    t=torch.linspace(-.9,.9,3*96*96).reshape(1,3,96,96);v=spatial(target_components(t),t,8)
    assert all(float(z)==0 for z in v.values());write_json(out/'objective_check.json',dict(perfect_region_spatial_loss={k:float(z) for k,z in v.items()}))


class Training(Prior):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        self.net=ComponentNet().cuda().eval();self.net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        aug=self.rng.random()<.5;original=x
        if aug:x=(x+torch.randn_like(x)*2/128).clamp(-1,127/128)
        p=self.net(x);y=straight_colors(p);losses=spatial(p,t,m)
        with torch.no_grad():
            xd=(original[1:]-warp(original[:1],flow)).abs().mean(1,keepdim=True);valid=mask*(xd<8/128);allowed=xd.clamp(max=3/128)+1/128
        delta=(y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True)
        losses['temporal']=(cut(F.relu(delta-allowed))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];q=self.net(shift)
        losses['phase']=(q[...,m+sy:-m,m+sx:-m]-p[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        loss=sum(v*self.cfg['objective_weights'][k] for k,v in losses.items());lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},grad=float(norm),lr=lr,native=native,noise=aug,sample=spec,seconds=time.perf_counter()-start)


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,macs=plan()['macs']));print('CENTER_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='component-centers-fp32');p.add_argument('--initial',type=Path);p.add_argument('--steps',type=int,default=3000);p.add_argument('--stop',type=int,default=3000);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100450);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':prepare(out,a.initial,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('CENTER_TRAIN',tr.step,round(row['loss'],4),round(row['center']*128,3),round(row['flat']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('CENTER_DONE',tr.step,flush=True)


if __name__=='__main__':main()
