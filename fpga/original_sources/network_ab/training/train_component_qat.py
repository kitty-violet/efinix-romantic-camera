"""Fixed-scale exact-Q17 forward, center objective, FP32 optimizer."""
import argparse,copy,gc,json
from pathlib import Path
import torch
from .train_component_centers import Training as CenterTraining
from .train_detail_rgb import Training as DataTraining
from .component_region_net import ComponentNet
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class Training(CenterTraining):
    def __init__(self,out):
        DataTraining.__init__(self,out,'qat');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256'];state=torch.load(p,map_location='cpu',weights_only=True)
        self.net=ComponentNet().cuda().eval();self.net.load_state_dict(state['model']);self.net.scales=state['scales'];self.calibration=state['calibration'];assert self.net.scales['rgb']==1/128
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['qat_lr'])


def prepare(out,initial,steps):
    state=torch.load(initial,map_location='cpu',weights_only=True);assert state['mode']=='component_ptq' and all(file_hash(k)==v for k,v in state['binding']['code'].items());source=Path(state['source']).parent;out.mkdir(exist_ok=False)
    for k in ['data.json','graph.json']:(out/k).write_bytes((source/k).read_bytes())
    cfg=copy.deepcopy(state['config']);cfg.update(seed=2026100452,initial=str(initial),initial_sha256=file_hash(initial),qat_steps=steps,fp32_steps=steps,qat_lr=.00008,fp32_lr=.00008,precision='FP32 optimizer, exact INT8/Q17 forward, no AMP/TF32/EMA',objective_config_alias='Inherited schedule uses fp32_steps/lr; equal to QAT schedule')
    write_json(out/'config.json',cfg);binding=copy.deepcopy(state['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__);binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False);expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False);err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,exact_integer_forward=True));print('COMPONENT_QAT_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='component-centers-qat');p.add_argument('--initial',type=Path);p.add_argument('--steps',type=int,default=1500);p.add_argument('--stop',type=int,default=1500);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100452);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':prepare(out,a.initial,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'qat_metrics.jsonl').exists()
    with (out/'qat_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['qat_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('COMPONENT_QAT',tr.step,round(row['loss'],4),round(row['center']*128,3),round(row['flat']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:
                torch.save(tr.state(),out/f'qat_checkpoint_{tr.step:06d}.pt');write_json(out/f'parameters_{tr.step:06d}.json',tr.net.integer_package());tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'qat_checkpoint_{tr.step:06d}.pt');return
    print('COMPONENT_QAT_DONE',tr.step,flush=True)


if __name__=='__main__':main()
