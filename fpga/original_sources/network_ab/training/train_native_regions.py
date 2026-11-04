"""Train a selective learned filter on the native detail branch."""
import argparse,gc,json
from pathlib import Path
import torch
from .train_hard_regions import Training as HardTraining
from .train_detail_rgb import Training as DataTraining
from .skip_filter_net import SkipFilterNet,plan
from .context_region_net import ContextNet
from .probe_context_skip import forward as diagnostic_forward
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out,initial,steps):
    root=ROOT/'runs'/RUN;audit=root/'native-context-budget48-v1/audit.json';budget=json.loads(audit.read_text())
    assert budget['status']=='FINITE_ALLOCATION_ONLY' and budget['source_sha256']==file_hash(Path(__file__).with_name('skip_filter_net.py'))
    state=torch.load(initial,map_location='cpu',weights_only=True);assert state['scales'] is None and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    out.mkdir(exist_ok=False);(out/'data.json').write_bytes((initial.parent/'data.json').read_bytes());write_json(out/'graph.json',plan())
    cfg=dict(state['config']);cfg.update(seed=2026100437,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=steps,fp32_lr=.0006,
        native_interval=100,budget_sha256=file_hash(audit),architecture='19 conv, learned native DW/PW before original additive skip',initial_native_filter='3x3 positive average followed by identity PW; learned thereafter')
    write_json(out/'config.json',cfg);binding=dict(state['binding']);binding['code']=dict(binding['code'])
    for f in [Path(__file__),Path(__file__).with_name('skip_filter_net.py'),Path(__file__).with_name('probe_context_skip.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(HardTraining):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        state=torch.load(p,map_location='cpu',weights_only=True);self.net=SkipFilterNet().cuda().eval();missing,extra=self.net.load_state_dict(state['model'],strict=False)
        assert not extra and missing and all(k.startswith('layers.native__') for k in missing)
        with torch.no_grad():self.net.conv('native.dw').weight.fill_(1/9)
        self.net.scales=None;self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])


def preflight(out):
    tr=Training(out);old=ContextNet().cuda().eval();old.load_state_dict(torch.load(tr.cfg['initial'],map_location='cpu',weights_only=True)['model'])
    with torch.no_grad():
        x=torch.rand(1,3,64,96,device='cuda')*2-1;err0=float((tr.net(x)-diagnostic_forward(old,x,3)).abs().max());assert err0<.0003,err0
    del old;tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,initializer_vs_diagnostic_error=err0,ideal_macs=plan()['macs']));print('NATIVE_PREFLIGHT_PASS',err0,err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='native-regions-fp32');p.add_argument('--initial',type=Path);p.add_argument('--steps',type=int,default=3000);p.add_argument('--stop',type=int,default=3000);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100437);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':assert a.initial;prepare(out,a.initial,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists();tr.evaluate();gc.collect();torch.cuda.empty_cache()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('NATIVE_TRAIN',tr.step,round(row['loss'],4),round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:
                torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('NATIVE_DONE',tr.step,flush=True)


if __name__=='__main__':main()
