"""Initialize a separately recorded stronger-style continuation and verify resume."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from . import coherent_shared_train as trainer
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text(encoding='utf-8'))
    init=Path(config['initial_checkpoint']);base=torch.load(init,map_location='cpu',weights_only=True)
    if base['step']!=config['prior_training_steps'] or base['config']['model']!=config['model']:
        raise ValueError('Initial checkpoint mismatch')
    code=[Path(trainer.__file__),Path(trainer.__file__).with_name('coherent_shared_model.py'),
          Path(trainer.__file__).with_name('losses.py'),Path(trainer.__file__).with_name('cut_wide_model.py')]
    contract=Path(r'E:\fpga\network_ab\model_b_contract.json')
    binding={'code':{str(p):file_hash(p) for p in code},'config_sha256':file_hash(a.config),
             'manifest_sha256':file_hash(config['manifest']),'hardware_contract_sha256':file_hash(contract),
             'role':'experimental_shared_teacher_not_hardware_contract_model'}
    if base['binding']['manifest_sha256']!=binding['manifest_sha256']:raise ValueError('Data changed')
    training=trainer.Training(config)
    training.net.load_state_dict(base['ema']);training.ema.load_state_dict(base['ema'])
    training.disc.load_state_dict(base['discriminator'])
    out=Path(config['preflight_run']);out.mkdir(parents=True,exist_ok=True)
    seed=out/'stage_initial.pt';torch.save(training.state(binding),seed)
    rows=[training.update() for _ in range(3)]
    saved=out/'preflight_resume.pt';torch.save(training.state(binding),saved)
    reference=training.update()
    expected={k:v.detach().cpu().clone() for k,v in training.net.state_dict().items()}
    training.restore(torch.load(saved,map_location='cpu',weights_only=True),binding)
    replay=training.update()
    error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in training.net.state_dict().items())
    with torch.no_grad():full=training.ema(torch.zeros(1,3,720,1280,device='cuda'))
    if error>1e-5 or not torch.isfinite(full).all():raise RuntimeError(f'Preflight failure {error}')
    result={'status':'PASS','binding':binding,'initial_checkpoint':str(init),
            'initial_checkpoint_sha256':file_hash(init),'initial_step':base['step'],
            'stage_initial':str(seed),'stage_initial_sha256':file_hash(seed),
            'steps':rows,'resume_reference':reference,'resume_replay':replay,
            'resume_max_abs_error':error,'full_720_shape':list(full.shape),
            'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2}
    write_json(out/'preflight.json',result)
    print(json.dumps({'status':'PASS','resume_max_abs_error':error,'stage_initial':str(seed)}))


if __name__=='__main__':main()
