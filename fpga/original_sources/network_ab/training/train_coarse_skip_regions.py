"""Region learning after removing the high-resolution texture bypass."""
import argparse,copy,gc,json
from pathlib import Path
import torch
from .train_intermediate_regions import Training as Previous
from .train_detail_rgb import Training as DataTraining
from .coarse_skip_net import CoarseSkipNet,plan
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out,initial):
    root=ROOT/'runs'/RUN;budget=root/'coarse-skip-budget64-v1/audit.json';a=json.loads(budget.read_text())
    assert a['status']=='FINITE_ALLOCATION_ONLY' and a['source_sha256']==file_hash(Path(__file__).with_name('coarse_skip_net.py'))
    state=torch.load(initial,map_location='cpu',weights_only=True);assert state['scales'] is None and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    out.mkdir(exist_ok=False);(out/'data.json').write_bytes((initial.parent/'data.json').read_bytes());write_json(out/'graph.json',plan())
    cfg=copy.deepcopy(state['config']);cfg.update(seed=2026100441,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=3000,fp32_lr=.0006,
        margin=48,native_interval=100,budget_path=str(budget),budget_sha256=file_hash(budget),
        architecture='three-scale DW/PW context; move skip to 360p; native 720p learned logits and fixed decoder',
        initialization='shared native-mid weights; new wide DW average/PW identity; down.dw average without bias')
    write_json(out/'config.json',cfg);binding=copy.deepcopy(state['binding'])
    for p in [Path(__file__),Path(__file__).with_name('coarse_skip_net.py'),Path(__file__).with_name('wide_context_net.py')]:binding['code'][str(p)]=file_hash(p)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(Previous):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        state=torch.load(p,map_location='cpu',weights_only=True);self.net=CoarseSkipNet().cuda().eval()
        weights={k:v for k,v in state['model'].items() if not k.startswith('layers.native__dw.')}
        missing,extra=self.net.load_state_dict(weights,strict=False);assert not extra and missing and all(k.startswith('layers.wide__') for k in missing)
        with torch.no_grad():self.net.conv('down.dw').weight.fill_(1/9);self.net.conv('down.dw').bias.zero_()
        self.net.scales=None;self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])


def preflight(out):
    tr=Training(out);calls=[];hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in tr.net.modules() if isinstance(m,torch.nn.Conv2d)]
    with torch.no_grad():y=tr.net(torch.zeros(1,3,720,1280,device='cuda'));assert y.shape==(1,32,720,1280)
    for h in hooks:h.remove()
    assert sum(calls)==plan()['macs'];tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,macs=sum(calls),convolutions=len(calls)));print('COARSE_SKIP_PREFLIGHT_PASS',err,sum(calls),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--initial',type=Path);p.add_argument('--stop',type=int,default=3000);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/'runs'/RUN/'coarse-skip-fp32'
    if a.action=='prepare':assert a.initial;prepare(out,a.initial);return
    configure_workspace();contract_guard();set_reproducible(2026100441)
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists();tr.evaluate();gc.collect();torch.cuda.empty_cache()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('COARSE_SKIP_TRAIN',tr.step,round(row['loss'],4),round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:
                torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('COARSE_SKIP_DONE',tr.step,flush=True)


if __name__=='__main__':main()
