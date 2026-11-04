"""General-scene continuation after controlled tying/quantization diagnosis.

Only existing training frames update the model. All published integer outputs
are regenerated from the saved B1 image; no image-space cleanup is applied.
"""
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from . import train_b1_structural_gate as gate
from .train_b1_deep_contour import DeepTraining, DeepStudent, OUT as PARENT
from .probe_b1_crop_quantization import calibrate_parameter_precision
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .b1_trained_image import decode, encode, runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

OUT = PARENT.parent / 'b1-general-contour-precision-20261005-v1'
PARENT_CHECKPOINT = PARENT / 'checkpoint_002000.pt'
FP32_STEPS, QAT_STEPS = 1000, 1000


def make(data):
    tr = DeepTraining(data)
    tr.net.load_state_dict(torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=True)['model'])
    tr.net.quantized = False
    tr.step = 0
    return tr


def save(tr, path):
    torch.save(dict(model=tr.net.state_dict(), optimizer=tr.opt.state_dict(),
                    runtime=tr.net.package, rng=tr.rng.getstate(), step=tr.step,
                    quantized=tr.net.quantized), path)


def restore(tr, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    tr.net.load_state_dict(state['model'])
    tr.net.package = state['runtime']
    tr.opt.load_state_dict(state['optimizer'])
    tr.rng.setstate(state['rng'])
    tr.step, tr.net.quantized = state['step'], state['quantized']


def gray(a):
    # Fixed gain and nearest replication make every internal cell visible.
    return Image.fromarray(np.clip(a * 255 / 24, 0, 255).astype(np.uint8)).resize(
        (1280, 720), Image.Resampling.NEAREST).convert('RGB')


def contact(rows, dest):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    for kind, columns in [('contours', ['source', 'thin', 'target', 'prediction']),
                          ('rgb', ['source', 'teacher', 'output'])]:
        if kind == 'rgb' and 'output' not in next(iter(rows.values())):
            continue
        board = Image.new('RGB', (420 * len(columns), 314 * len(gate.CROPS)), '#17202a')
        draw = ImageDraw.Draw(board)
        titles = dict(source='原片', thin='细线标签：不是模型结果',
                      target='内部训练信号：同网格', prediction='实际小网轮廓响应',
                      teacher='冻结彩色目标', output='真实权重重载 RGB')
        for i, (key, box, name) in enumerate(gate.CROPS):
            for j, col in enumerate(columns):
                board.paste(rows[key][col].crop(box), (420*j, 314*i+34))
                draw.text((420*j+5, 314*i+4), titles[col] + (' '+name if j == 0 else ''), font=font, fill='white')
        board.save(dest / ('native_' + kind + '.png'))


@torch.no_grad()
def review(tr, data, tag, integer=False, target_fn=None):
    dest = OUT / tag
    dest.mkdir(exist_ok=False)
    previous = tr.net.quantized
    tr.net.quantized = integer
    actual = None
    if integer:
        raw = tr.net.export()
        assert len(raw) == 20480
        (dest/'coefficients-padded.bin').write_bytes(raw)
        params, residual = decode(raw)
        assert encode(params, residual) == raw[:17792]
        actual = IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
    rows, metrics, node_errors = {}, [], {}
    for r in data:
        if r['split'] == 'train':
            continue
        rgb = np.array(Image.open(r['source']).convert('RGB'))
        x = torch.tensor(rgb.transpose(2, 0, 1)[None].copy(), device='cuda', dtype=torch.float32)/128-1
        if integer:
            nodes, _ = actual(x*128, True)
            learned = tr.net(x, True, True)
            error = max(float((v-nodes[k]).abs().max()) for k,v in learned.items())
            assert error == 0
            node_errors[r['id']] = error
            pred = nodes[tr.net.first_gate_node][0,24].cpu().numpy()
            assert torch.equal(nodes[tr.net.first_gate_node][:,24:25], nodes[tr.net.rectified_gate_node][:,24:25])
            output = Image.fromarray((nodes['rgb_head'][0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8))
            output.save(dest/(r['id']+'_raw_rgb.png'))
        else:
            pred = tr.net(x)[0,0].cpu().numpy().clip(0)
        target = (np.maximum(np.load(r['label']).astype(np.float32)[::4,::4]*32-8, 0)
                  if target_fn is None else target_fn(r))
        valid = np.zeros_like(target, dtype=bool); valid[8:-8,8:-8] = True
        positive, truth = (pred > 4)&valid, (target > 4)&valid
        empty = (target == 0)&valid
        metrics.append(dict(id=r['id'], precision=float((positive&truth).sum()/max(positive.sum(),1)),
                            recall=float((positive&truth).sum()/max(truth.sum(),1)),
                            mae=float(abs(pred-target)[valid].mean()),
                            false_response_on_zero_target=float(((pred>4)&empty).sum()/max(empty.sum(),1))))
        prediction = gray(pred)
        prediction.save(dest/(r['id']+'_gate.png'))
        rows[r['id']] = dict(source=Image.fromarray(rgb), teacher=Image.open(r['target']).convert('RGB'),
                             thin=Image.open(PARENT/'labels'/(r['id']+'_thin.png')).convert('RGB'),
                             target=gray(target), prediction=prediction)
        if integer:
            rows[r['id']]['output'] = output
    independent_error = None
    if integer:
        op, rp = oracle_records(params)
        contract, _ = contract_guard()
        sample = np.random.default_rng(31).integers(-128,128,(12,12,3),dtype=np.int16)
        scalar = execute_integer(contract, sample.tolist(), op, rp)
        tensors, _ = actual(torch.tensor(sample.transpose(2,0,1)[None], device='cuda',dtype=torch.float32),True)
        independent_error = max(int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(scalar['tensors'][k])).max()) for k,v in tensors.items())
        assert independent_error == 0
    contact(rows, dest)
    write_json(dest/'verification.json', dict(step=tr.step, integer=integer, metrics=metrics,
        every_scene_node_error=node_errors, independent_oracle_error=independent_error,
        weight_sha256=file_hash(dest/'coefficients-padded.bin') if integer else None,
        reviewed_regression_not_blind=True, visual_accepted=False, no_external_cleanup=True))
    tr.net.quantized = previous
    print('GENERAL_REVIEW', tag, metrics, flush=True)


def preflight(tr, data):
    dest = OUT/'preflight'; dest.mkdir()
    rgb = np.array(Image.open(data[0]['source']).convert('RGB'))[:128,:128]
    x = torch.tensor(rgb.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)/128-1
    errors = {}
    for quantized in (False, True):
        tr.net.quantized = quantized
        tr.net.zero_grad(set_to_none=True)
        fast = tr.net(x); fast.square().mean().backward()
        gradients = {k: p.grad.clone() for k,p in tr.net.named_parameters() if p.grad is not None}
        tr.net.zero_grad(set_to_none=True)
        complete = tr.net(x, True, True)['res2_add'][:,24:25]
        complete.square().mean().backward()
        pe = float((fast-complete).abs().max())
        ge = max(float((p.grad-gradients[k]).abs().max()) for k,p in tr.net.named_parameters() if k in gradients)
        assert pe == 0 and ge == 0
        errors[str(quantized)] = dict(forward=pe, gradient=ge)
    tests=[]
    for quantized in (False, True):
        tr.net.quantized=quantized
        if quantized:
            calibrate_parameter_precision(tr.net)
        tests.append(tr.update(True))
        tests.append(tr.update())
        save(tr, dest/('qat_resume.pt' if quantized else 'fp32_resume.pt'))
        expected_record = tr.update()
        expected = {k:v.detach().clone() for k,v in tr.net.state_dict().items()}
        fresh = make(data)
        restore(fresh,dest/('qat_resume.pt' if quantized else 'fp32_resume.pt'))
        record = fresh.update()
        error = max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
        assert error==0 and record['crop']==expected_record['crop'] and record['sample']==expected_record['sample']
        errors['resume_'+str(quantized)] = error
        del fresh
    write_json(dest/'result.json',dict(status='PASS',fast_full_equivalence=errors,native_and_crop_tests=tests))
    print('GENERAL_PREFLIGHT_PASS',errors,flush=True)


def main():
    configure_workspace();set_reproducible(2026100509);contract_guard()
    OUT.mkdir(exist_ok=False)
    data=json.loads((PARENT/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):
            assert file_hash(r[key])==r[key+'_sha256']
    dependencies=[Path(__file__),Path(gate.__file__),Path(__file__).with_name('train_b1_deep_contour.py'),
                  Path(__file__).with_name('train_b1_structural_features.py'),Path(__file__).with_name('probe_b1_crop_quantization.py')]
    for p in dependencies:shutil.copy2(p,OUT/p.name)
    write_json(OUT/'binding.json',dict(parent=str(PARENT_CHECKPOINT),parent_sha256=file_hash(PARENT_CHECKPOINT),
        data_sha256=file_hash(PARENT/'data.json'),code={p.name:file_hash(p) for p in dependencies},
        fp32_steps=FP32_STEPS,qat_steps=QAT_STEPS,scope='General training continuation, frozen colour lanes, contour first'))
    tr=make(data);preflight(tr,data)
    del tr;torch.cuda.empty_cache();tr=make(data)
    review(tr,data,'before_fp32')
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<FP32_STEPS+QAT_STEPS:
            if tr.step==FP32_STEPS:
                changes=calibrate_parameter_precision(tr.net)
                write_json(OUT/'parameter_calibration.json',changes)
                tr.net.quantized=True
                tr.opt=torch.optim.Adam([p for p in tr.net.parameters() if p.requires_grad],lr=.015)
                review(tr,data,'calibrated_integer',True)
            if tr.step<FP32_STEPS:
                lr=.018-.010*(tr.step/FP32_STEPS)
            else:
                phase=(tr.step-FP32_STEPS)/QAT_STEPS
                lr=.003+(.015-.003)*.5*(1+math.cos(math.pi*phase))
            for group in tr.opt.param_groups:group['lr']=lr
            row=tr.update((tr.step+1)%100==0);row['lr']=lr;log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('GENERAL_TRAIN',tr.step,round(row['loss'],4),round(row['seconds'],3),'QAT',tr.net.quantized,flush=True)
            if tr.step%500==0:
                save(tr,OUT/f'checkpoint_{tr.step:06d}.pt')
                review(tr,data,f'step_{tr.step:06d}',tr.net.quantized)
    print('GENERAL_COMPLETE',flush=True)


if __name__=='__main__':main()
