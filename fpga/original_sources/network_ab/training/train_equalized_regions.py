"""QAT continuation after a verified function-preserving reparameterization."""
import argparse,gc,json
from pathlib import Path
import torch
from .train_luma_regions import Training as LumaTraining,ROOT,RUN
from .equalize_luma import EqualizedNet
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

def prepare(out):
    root=ROOT/'runs'/RUN;out.mkdir(exist_ok=False);base=root/'luma'
    (out/'data.json').write_bytes((base/'data.json').read_bytes());cfg=json.loads((base/'config.json').read_text())
    initial=root/'equalized-v2/equalized.pt';cfg.update(seed=2026100429,qat_steps=1500,qat_lr=.00015,initial=str(initial),initial_sha256=file_hash(initial))
    write_json(out/'config.json',cfg)
    binding=json.loads((base/'binding.json').read_text());binding['code'].update({str(p):file_hash(p) for p in (Path(__file__),Path(__file__).with_name('equalize_luma.py'))})
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'))
    write_json(out/'binding.json',binding)

class Training(LumaTraining):
    def __init__(self,out):
        super().__init__(out,'qat');self.net=EqualizedNet().cuda().eval();p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        s=torch.load(p,map_location='cpu',weights_only=True);self.net.load_state_dict(s['model']);self.net.scales=s['scales'];self.calibration=s['calibration']
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['qat_lr'])

def preflight(out):
    tr=Training(out);tr.update(True);tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample']
    write_json(out/'preflight.json',dict(status='PASS',full720_pair_forward_backward=True,batch2=True,checkpoint_restore_error=err));print('EQ_PREFLIGHT_PASS',err,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--stop',type=int,default=1500);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100429);out=ROOT/'runs'/RUN/'equalized-qat'
    if a.action=='prepare':prepare(out);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'qat_metrics.jsonl').exists();tr.evaluate()
    write_json(out/f'launch_{tr.step:06d}.json',dict(start=tr.step,resume=str(a.resume),binding=tr.binding))
    with (out/'qat_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['qat_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('EQ_QAT',tr.step,round(row['loss'],4),round(row['phase'],5),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'qat_checkpoint_{tr.step:06d}.pt');tr.evaluate()
    print('EQ_SEGMENT_DONE',tr.step,flush=True)

if __name__=='__main__':main()
