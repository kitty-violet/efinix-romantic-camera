"""Teach texture suppression with paired texture changes inside target regions.

Training-only curriculum. Inference remains the identical learned FPGA graph.
Half the real adjacent pairs retain their original inputs. The other half use
the region target plus controllable fine source texture and sensor noise; their
target/geometry/flow do not change. No validation samples enter this process.
"""
import argparse,copy,gc,json
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_ordinal_regions import Training as Previous
from .train_detail_rgb import Training as DataTraining
from .ordinal_coarse_net import OrdinalCoarseNet
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out):
    initial=ROOT/'runs'/RUN/'ordinal-regions-fp32/fp32_checkpoint_002000.pt';state=torch.load(initial,map_location='cpu',weights_only=True)
    assert all(file_hash(k)==v for k,v in state['binding']['code'].items());out.mkdir(exist_ok=False)
    for f in ['data.json','graph.json']:(out/f).write_bytes((initial.parent/f).read_bytes())
    cfg=copy.deepcopy(state['config']);cfg.update(seed=2026100448,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=3000,fp32_lr=.0005,
        curriculum_probability=.5,source_texture_gain=[.5,1.5],synthetic_noise_rgb255=[0,8],curriculum='target regions + source local fine texture; paired adjacent geometry, unchanged real targets',
        objective_weights=dict(ce=.5,color=3,flat=32,region=2,surface=8,temporal=3,phase=3))
    write_json(out/'config.json',cfg);binding=copy.deepcopy(state['binding']);binding['code'][str(Path(__file__))]=file_hash(__file__)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(Previous):
    def batch(self,native):
        x,t,flow,mask,spec=DataTraining.batch(self,native);self.curriculum=self.rng.random()<self.cfg['curriculum_probability']
        if self.curriculum:
            gain=self.rng.uniform(*self.cfg['source_texture_gain']);noise=self.rng.uniform(*self.cfg['synthetic_noise_rgb255'])
            coarse=F.avg_pool2d(F.pad(x,(3,3,3,3),mode='replicate'),7,stride=1);fine=x-coarse
            x=(t+gain*fine+torch.randn_like(x)*(noise/128)).clamp(-1,127/128)
            spec=[*spec,'paired_texture_curriculum',gain,noise]
        return x,t,flow,mask,spec

    def update(self,native=None):
        row=super().update(native);row['curriculum']=self.curriculum;return row


def preflight(out):
    tr=Training(out);tr.cfg['curriculum_probability']=1.;a0=tr.update(True);tr.cfg['curriculum_probability']=0.;a1=tr.update(False)
    tr.cfg['curriculum_probability']=.5;torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,curriculum_full720=a0['curriculum'],real_pair_batch2=not a1['curriculum']));print('CURRICULUM_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--stop',type=int,default=3000);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/'runs'/RUN/'surface-curriculum-fp32'
    if a.action=='prepare':prepare(out);return
    configure_workspace();contract_guard();set_reproducible(2026100448)
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('SURFACE_CURRICULUM',tr.step,round(row['loss'],4),row['curriculum'],round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:tr.save();tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():tr.save();return
    print('SURFACE_CURRICULUM_DONE',tr.step,flush=True)


if __name__=='__main__':main()
