"""Use the preceding frame as a fixed temporal target for deterministic updates.

Only the training loss changes: forward warping is unchanged, gradients go to
the current frame. Spatial supervision still trains BOTH real adjacent frames.
The deployed convolution graph is identical to color-input-v2.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import cv2
import torch
from . import train_stroke_color as base
from .train_stroke_color_input import Training
from .stroke_color_input_net import StrokeColorInputNet
from .coherent_shared_model import warp as original_warp
from .engine import set_reproducible
from .runtime import configure_workspace,contract_guard,file_hash,write_json


def reference_warp(image,backward):
    with torch.no_grad():return original_warp(image,backward)


def prepare(out,steps):
    source=base.ROOT/'native-stroke-color-input-v2';out.mkdir(exist_ok=False)
    for name in ('data.json','color_initial.pt','initialization.json'):shutil.copy2(source/name,out/name)
    shutil.copytree(source/'source',out/'source')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    cfg.update(initial=str(out/'color_initial.pt'),initial_sha256=file_hash(out/'color_initial.pt'),steps=steps,
        temporal='preceding predicted frame is detached before warp; current prediction receives temporal gradient; both frames have spatial supervision')
    write_json(out/'config.json',cfg)
    b=json.loads((source/'binding.json').read_text())
    assert all(file_hash(p)==h for p,h in b['code'].items()) and all(file_hash(p)==h for p,h in b['images'].items())
    b['code'][str(Path(__file__))]=file_hash(__file__);shutil.copy2(__file__,out/'source'/Path(__file__).name)
    b['config']=file_hash(out/'config.json');b['data']=file_hash(out/'data.json');write_json(out/'binding.json',b)
    # Preserve the interrupted trial as evidence; do not reuse its uncheckpointed updates.
    lines=(source/'metrics.jsonl').read_text().splitlines()
    write_json(source/'status.json',dict(status='STOPPED_ACTIVE_TEMPORAL_REPLAY_FAILED',logged_updates=len(lines),last_step=json.loads(lines[-1])['step'],
        reason='CUDA grid_sample backward replay parameter difference 1.1324882507324219e-06 > 1e-6; no final weights published'))
    print('STABLE_COLOR_PREPARED',flush=True)


def active_check(out):
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    t=Training(out);t.step=500;t.update(native=False)
    torch.save(t.state(),out/'active_temporal_preflight.pt');ref=t.update(native=False);expected=copy.deepcopy(t.net.state_dict())
    t.restore(torch.load(out/'active_temporal_preflight.pt',map_location='cpu',weights_only=True));replay=t.update(native=False)
    error=max(float((v-expected[k]).abs().max()) for k,v in t.net.state_dict().items())
    passed=error<=1e-7 and ref['sample']==replay['sample']
    write_json(out/'active_temporal_resume.json',dict(status='PASS' if passed else 'FAIL',resume_max_error=error,reference=ref,replay=replay,
        diagnostic_schedule_step=500,actual_formal_training_steps=0,binding=t.binding,
        temporal_reference_detached=True,temporal_forward_unchanged=True))
    assert passed
    print('ACTIVE_TEMPORAL_RESTORE_PASS',error,flush=True)


def main():
    base.StrokeColorNet=StrokeColorInputNet;base.Training=Training;base.warp=reference_warp;base.prepare=prepare
    if '--name' not in sys.argv:sys.argv.extend(['--name','native-stroke-color-stable-v3'])
    out=base.ROOT/sys.argv[sys.argv.index('--name')+1]
    if sys.argv[1]=='active-check':active_check(out);return
    if sys.argv[1]=='train':
        gate=json.loads((out/'active_temporal_resume.json').read_text());assert gate['status']=='PASS'
        assert gate['binding']==json.loads((out/'binding.json').read_text())
    base.main()


if __name__=='__main__':main()
