"""Same hard-region objective with exact integer-forward weights/scales."""
import argparse,gc,json
from pathlib import Path
import torch
from .train_intermediate_regions import Training as IntermediateTraining
from .train_detail_rgb import Training as DataTraining
from .equalize_native import EqualizedNative
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out,initial,steps):
    state=torch.load(initial,map_location='cpu',weights_only=True);source=Path(state['source']).parent
    assert state['mode']=='equalized_ptq' and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    out.mkdir(exist_ok=False)
    for k in ['data.json','graph.json']:(out/k).write_bytes((source/k).read_bytes())
    cfg=dict(state['config']);cfg.update(seed=2026100440,initial=str(initial),initial_sha256=file_hash(initial),
        qat_steps=steps,qat_lr=.00010,fp32_steps=steps,fp32_lr=.00010,
        objective_config_alias='HardTraining.update uses fp32_steps/lr names; these equal QAT schedule here',precision='FP32 optimizer, Q17 exact integer-forward; no AMP/TF32/EMA')
    write_json(out/'config.json',cfg);binding=dict(state['binding']);binding['code']=dict(binding['code']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(IntermediateTraining):
    def __init__(self,out):
        DataTraining.__init__(self,out,'qat');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        state=torch.load(p,map_location='cpu',weights_only=True);self.net=EqualizedNative().cuda().eval();self.net.load_state_dict(state['model']);self.net.scales=state['scales'];self.calibration=state['calibration']
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['qat_lr'])


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,integer_forward=True));print('NATIVE_MID_QAT_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='native-mid-qat');p.add_argument('--initial',type=Path);p.add_argument('--steps',type=int,default=1500);p.add_argument('--stop',type=int,default=1500);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100440);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':assert a.initial;prepare(out,a.initial,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'qat_metrics.jsonl').exists();tr.evaluate();gc.collect();torch.cuda.empty_cache()
    with (out/'qat_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['qat_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('NATIVE_MID_QAT',tr.step,round(row['loss'],4),round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:
                torch.save(tr.state(),out/f'qat_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'qat_checkpoint_{tr.step:06d}.pt');return
    print('NATIVE_MID_QAT_DONE',tr.step,flush=True)


if __name__=='__main__':main()
