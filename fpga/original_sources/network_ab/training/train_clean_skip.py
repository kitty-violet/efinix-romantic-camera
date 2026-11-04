"""Train-only clean-structure methods with filtering capacity on the skip."""
import argparse,copy,gc,json,time
from pathlib import Path
import torch
from . import train_clean_capacity as method
from .train_detail_rgb import Training as DataTraining
from .b1_capacity_component_net import CapacityComponentNet
from .clean_skip_capacity_net import CleanSkipNet,plan,budget
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class Training(method.Training):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        self.net=CleanSkipNet().cuda().eval();self.net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])
        self.reference=CleanSkipNet().cuda().eval();self.reference.load_state_dict(self.net.state_dict());self.reference.requires_grad_(False)
        self.critic=method.TextureCritic().cuda().train();self.dopt=torch.optim.Adam(self.critic.parameters(),lr=self.cfg['critic_lr'],betas=(.5,.99))


def prepare(out,steps):
    method.prepare(out,steps)
    # This earlier widened state has EXACT identity extra PW layers, allowing
    # a function-preserving relocation and avoiding another random color start.
    source=ROOT/'runs'/RUN/'b1-capacity-control-v2/capacity/initial.pt';s=torch.load(source,map_location='cpu',weights_only=True)
    assert file_hash(s['source'])==s['source_sha256'];old=CapacityComponentNet().eval();old.load_state_dict(s['model']);new=CleanSkipNet().eval();new.initialize(old)
    with torch.no_grad():
        x=torch.rand(2,3,96,160)*1.9-.95;error=float((new(x)-old(x)).abs().max());assert error<2e-6
    initial=out/'initial.pt';torch.save(dict(model=new.state_dict(),source=str(source),source_sha256=file_hash(source)),initial)
    cfg=json.loads((out/'config.json').read_text());cfg.update(initial=str(initial),initial_sha256=file_hash(initial),
        feature_nodes=['native.texture.pw','down.pw'],inference_changed=True,
        method='Move two low-res PW layers into 360p skip DW/PW; same clean-feature/texture/temporal objective',
        source_model_note='Function-preserving relocation from the same pre-expansion-training generator',graph_schema='clean-skip-capacity-v1')
    write_json(out/'config.json',cfg);write_json(out/'graph.json',plan());binding=json.loads((out/'binding.json').read_text())
    for f in [Path(__file__),Path(__file__).with_name('clean_skip_capacity_net.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    checks=json.loads((out/'objective_checks.json').read_text());checks.update(budget=budget(),initial_function_error=error);write_json(out/'objective_checks.json',checks)
    print('CLEAN_SKIP_PREPARED',budget()['trainable_parameters'],budget()['ideal_macs'],error,flush=True)


def preflight(out):
    old=method.Training;old_budget=method.budget;method.Training=Training;method.budget=budget
    try:method.preflight(out)
    finally:method.Training=old;method.budget=old_budget
    n=CleanSkipNet().cuda().eval();calls=[];hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in n.modules() if isinstance(m,torch.nn.Conv2d)]
    with torch.no_grad():y=n(torch.zeros(1,3,720,1280,device='cuda'))
    for h in hooks:h.remove()
    assert sum(calls)==plan()['macs'] and y.shape==(1,3,720,1280)
    r=json.loads((out/'preflight.json').read_text());r.update(actual_forward_macs=sum(calls),shape=list(y.shape),graph_changed=True);write_json(out/'preflight.json',r)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='clean-skip-v1');p.add_argument('--steps',type=int,default=2000);p.add_argument('--stop',type=int,default=500);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100461);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':prepare(out,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists();tr.evaluate()
    write_json(out/f'launch_{tr.step:06d}.json',dict(start=tr.step,stop=a.stop,graph_changed=True,budget=budget(),bindings=tr.binding,preflight='PASS'))
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('CLEAN_SKIP_TRAIN',tr.step,round(row['loss'],4),round(row['clean']*128,4),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('CLEAN_SKIP_DONE',tr.step,flush=True)


if __name__=='__main__':main()
