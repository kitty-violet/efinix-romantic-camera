"""Source-to-itself reconstruction with the unchanged deployed B1 operators.

No teacher, contour labels, colour levels, external filtering or high-res skip.
All convolution coefficients/biases train; FP32 and true Q17/INT8 are separated.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from . import train_b1_learned_colour as c
from .audit_b1_sampling_limits import identity_image
from .b1_trained_image import decode, runtime_from_image
from .engine import set_reproducible
from .engineer_b1_blocks import oracle_records
from .probe_b1_cartoon_int8 import IntegerB1
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from network_ab.contract_reference import execute_integer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'model-training-workspace/runs/b1-identity-reconstruction-20261005'
MANIFEST = ROOT/'model-training-workspace/runs/b1-distillation-20261003/data/manifest.json'
CONTRACT_SHA = '5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
SEED = 2026100551


@lru_cache(maxsize=192)
def read(path):
    a = np.array(Image.open(path).convert('RGB'))
    assert a.shape == (720, 1280, 3)
    return a


def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().cuda()/128-1


def data():
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    train = {}
    for p in m['pairs']:
        assert m['source_splits'][p['source']]['split'] == 'train'
        for key in ('current', 'next'):
            path = p[key]
            assert file_hash(path) == p[key+'_sha256']
            train[path] = dict(id=Path(path).stem, path=path, sha256=p[key+'_sha256'],
                               video=p['source'], split='train', target_same_as_input=True)
    validation = []
    for row in m['validation']:
        assert row['raw_source'] not in {r['video'] for r in train.values()}
        assert file_hash(row['source']) == row['source_sha256']
        validation.append(dict(id=row['id'], path=row['source'], sha256=row['source_sha256'],
                               video=row['raw_source'], split='reviewed_validation', target_same_as_input=True))
    assert not ({r['sha256'] for r in train.values()} & {r['sha256'] for r in validation})
    return list(train.values()), validation


def initialize(seed=SEED):
    torch.manual_seed(seed)
    net = c.ColourB1(revive=False, source=OUT/'initial_identity.bin').cuda().eval()
    # Every residual path must have a live derivative. Zeroing all four layers
    # (as in the diagnostic identity) would permanently disable those branches.
    with torch.no_grad():
        for key, w in net.weight.items():
            if key.startswith('res'):
                w.zero_()
                if key.endswith('__dw'):
                    w[:, 0, 1, 1] = 1
                elif key.endswith('conv1__pw'):
                    for i in range(len(w)):
                        w[i, i, 0, 0] = 1
                else:
                    w.normal_(0, .0001)
            elif key != 'rgb_head':
                w.add_(torch.randn_like(w)*.001)
        for p in net.parameters():
            p.requires_grad_(True)
    net.quantized = False
    assert all(p.requires_grad for p in net.parameters())
    return net


class Trainer:
    def __init__(self, train, max_steps, qat=False, edge_weight=0.0):
        self.net = initialize()
        self.rows = train
        self.groups = {}
        for row in train:
            self.groups.setdefault(row['video'], []).append(row)
        self.rng = random.Random(SEED)
        self.step = 0
        self.max_steps = max_steps
        self.qat = qat
        self.edge_weight = edge_weight
        if qat:
            self.net.calibrate()
            self.net.quantized = True
        self.opt = torch.optim.Adam([
            dict(params=list(self.net.weight.values()), lr=.00035 if not qat else .00008),
            dict(params=list(self.net.bias.values()), lr=.012 if not qat else .004)])
        self.base_lrs = [g['lr'] for g in self.opt.param_groups]

    def batch(self, native):
        specs, images = [], []
        for _ in range(1 if native else 2):
            row = self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))])
            h, w = (720, 1280) if native else (320, 320)
            y, x = (0, 0) if native else (self.rng.randrange(721-h), self.rng.randrange(1281-w))
            a = read(row['path'])[y:y+h, x:x+w]
            flip = self.rng.randrange(2) if not native else 0
            if flip:
                a = a[:, ::-1]
            specs.append(dict(id=row['id'], xy=[x, y], wh=[w, h], flip=flip))
            images.append(tensor(a))
        return torch.stack(images), specs

    def update(self, native=None):
        begin = time.perf_counter()
        native = (self.step+1) % 100 == 0 if native is None else native
        x, specs = self.batch(native)
        p = self.net(x)[0]
        p, target = (p, x) if native else (p[:, :, 40:-40, 40:-40], x[:, :, 40:-40, 40:-40])
        mse = F.mse_loss(p, target)
        edge = sum(F.l1_loss(torch.diff(p, dim=d), torch.diff(target, dim=d)) for d in (-1, -2))/2
        loss = mse + self.edge_weight*edge
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5, error_if_nonfinite=True)
        assert torch.isfinite(loss) and grad > 0
        factor = .15 + .85*.5*(1+math.cos(math.pi*min(self.step/self.max_steps, 1)))
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group['lr'] = base*factor
        self.opt.step()
        self.step += 1
        torch.cuda.synchronize()
        return dict(step=self.step, mse=float(mse.detach()), edge=float(edge.detach()),
                    loss=float(loss.detach()), grad=float(grad), native720=native, samples=specs,
                    seconds=time.perf_counter()-begin)

    def save(self, path):
        contract_guard()
        torch.save(dict(model=self.net.state_dict(), optimizer=self.opt.state_dict(), step=self.step,
            max_steps=self.max_steps, qat=self.qat, edge_weight=self.edge_weight,
            rng=self.rng.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            contract_sha256=CONTRACT_SHA, target='input_itself', hardware_changed=False), path)

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['contract_sha256'] == file_hash(ROOT/'network_ab/model_b_contract.json') == CONTRACT_SHA
        assert state['max_steps'] == self.max_steps and state['qat'] == self.qat
        self.net.load_state_dict(state['model'])
        self.opt.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def preflight(train, dest, steps, qat=False, state=None):
    dest.mkdir(exist_ok=True)
    tr = Trainer(train, steps, qat)
    if state is not None:
        tr.net.load_state_dict(state)
    torch.cuda.reset_peak_memory_stats()
    native = tr.update(True)
    crop = tr.update(False)
    tr.save(dest/'resume_probe.pt')
    expected_step = tr.update(False)
    expected = {k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh = Trainer(train, steps, qat)
    fresh.restore(dest/'resume_probe.pt')
    actual_step = fresh.update(False)
    error = max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error < 1e-5 and expected_step['samples'] == actual_step['samples']
    write_json(dest/'preflight.json', dict(status='PASS', native=native, crop=crop,
        resume_max_error=error, peak_cuda_bytes=torch.cuda.max_memory_allocated(), qat=qat,
        target_exactly_input=True, trainable_coefficients=sum(p.numel() for p in tr.net.parameters())))
    print('RECON_PREFLIGHT_PASS', qat, error, native['seconds'], flush=True)
    del tr, fresh, expected
    torch.cuda.empty_cache()


def metrics(pred, target):
    delta = (pred-target)*128
    mse = float(delta.square().mean())
    edge_error = sum(float((torch.diff(pred,dim=d)-torch.diff(target,dim=d)).abs().mean()*128) for d in (-1,-2))/2
    return dict(mse_rgb255=mse, psnr=10*math.log10(255**2/max(mse,1e-12)),
                mae=float(delta.abs().mean()), edge_mae=edge_error)


@torch.inference_mode()
def evaluate(net, rows):
    records = []
    for row in rows:
        x = tensor(read(row['path']))[None]
        p = net(x)[0]
        records.append(dict(id=row['id'], split=row['split'], **metrics(p,x)))
    mean = sum(r['mse_rgb255'] for r in records)/len(records)
    return dict(records=records, pooled_psnr=10*math.log10(255**2/mean), mean_mse=mean)


@torch.inference_mode()
def preview(net, rows, dest):
    dest.mkdir(exist_ok=True)
    byid = {r['id']:r for r in rows}
    pictures = {}
    for key, _, _ in c.CROPS:
        if key not in pictures:
            a = read(byid[key]['path'])
            p = c.rgb_image(net(tensor(a)[None])[0][0])
            pictures[key] = (Image.fromarray(a), p)
            p.save(dest/(key+'_reconstruction.png'))
    board = Image.new('RGB',(840,314*len(c.CROPS)),'#17202a')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for i,(key,box,label) in enumerate(c.CROPS):
        for j,im in enumerate(pictures[key]):
            board.paste(im.crop(box),(420*j,314*i+34))
            draw.text((420*j+4,314*i+5),label+' | '+('原图（720p）' if j==0 else 'B1原图重建'),font=font,fill='white')
    board.save(dest/'comparison.png')


def train_stage(tr, train, val, dest, min_steps, interval=500, patience=3):
    dest.mkdir(exist_ok=True)
    initial = evaluate(tr.net,val)
    write_json(dest/'initial.json',initial)
    preview(tr.net,val,dest/'initial')
    history = [dict(step=0,validation=initial)]
    best = initial['pooled_psnr']
    tr.save(dest/'best.pt')
    significant = best
    flat = 0
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step < tr.max_steps:
            row = tr.update()
            log.write(json.dumps(row)+'\n')
            if tr.step % 100 == 0:
                print('RECON_TRAIN',dest.name,tr.step,round(row['mse'],6),round(row['seconds'],3),flush=True)
            if tr.step % interval == 0 or tr.step == tr.max_steps:
                score = evaluate(tr.net,val)
                train_score = evaluate(tr.net,train[::max(1,len(train)//4)][:4])
                history.append(dict(step=tr.step, validation=score, train=train_score))
                if score['pooled_psnr'] > best:
                    best = score['pooled_psnr']
                    tr.save(dest/'best.pt')
                    preview(tr.net,val,dest/'best')
                if score['pooled_psnr'] > significant+.03:
                    significant = score['pooled_psnr']; flat = 0
                else:
                    flat += 1
                tr.save(dest/'last.pt')
                write_json(dest/'history.json',history)
                write_json(dest/'progress.json',dict(step=tr.step, best_psnr=best, current_psnr=score['pooled_psnr'], flat_evaluations=flat))
                print('RECON_EVAL',dest.name,tr.step,'PSNR',score['pooled_psnr'],'BEST',best,'FLAT',flat,flush=True)
                if tr.step >= min_steps and flat >= patience:
                    break
    write_json(dest/'completion.json',dict(completed_steps=tr.step,best_psnr=best,
        stop='validation_plateau' if tr.step<tr.max_steps else 'bounded_budget',
        theoretical_upper_bound_proven=False, hardware_changed=False))
    return history


@torch.inference_mode()
def export_review(net, val, dest):
    dest.mkdir(exist_ok=True)
    assert net.quantized
    raw = net.export()
    (dest/'coefficients-padded.bin').write_bytes(raw)
    assert len(raw)==20480
    (dest/'load_words.hex').write_text(''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n' for i in range(0,len(raw),8)),encoding='ascii')
    actual = IntegerB1(runtime_from_image(raw))
    records = []
    for row in val:
        x = tensor(read(row['path']))[None]
        nodes,_ = actual(x*128,True)
        ours = net(x,all_nodes=True)
        error = max(float((v-nodes[k]).abs().max()) for k,v in ours.items())
        assert error == 0
        records.append(dict(id=row['id'],export_node_error=error,**metrics(nodes['rgb_head']/128,x)))
    params,_ = decode(raw)
    op,rp = oracle_records(params)
    sample = np.random.default_rng(51).integers(-128,128,(12,12,3),dtype=np.int16)
    contract,_ = contract_guard()
    ref = execute_integer(contract,sample.tolist(),op,rp)
    got,_ = actual(torch.tensor(sample.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32),True)
    error = max(float(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in got.items())
    assert error == 0
    write_json(dest/'review.json',dict(integer=True,independent_oracle_error=error,records=records,
        package_sha256=file_hash(dest/'coefficients-padded.bin'),purpose='identity_reconstruction_not_style',actual_board=False))
    preview(net,val,dest)
    print('RECON_EXPORT_VERIFIED',flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=8000)
    ap.add_argument('--qat',type=int,default=2000)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--continue-after-preflight',action='store_true')
    a = ap.parse_args()
    configure_workspace();set_reproducible(SEED);contract_guard()
    assert file_hash(ROOT/'network_ab/model_b_contract.json') == CONTRACT_SHA
    train,val = data()
    if a.continue_after_preflight:
        assert json.loads((OUT/'preflight/preflight.json').read_text())['status']=='PASS'
        assert not (OUT/'fp32').exists()
    else:
        OUT.mkdir(exist_ok=False)
        (OUT/'initial_identity.bin').write_bytes(identity_image())
        write_json(OUT/'data.json',dict(training=train,validation=val,training_videos=len({r['video'] for r in train}),
            evaluation_note='Source-disjoint from training; six previously reviewed frames, not a blind test.',
            parent_manifest=str(MANIFEST),parent_sha256=file_hash(MANIFEST),target='source_itself'))
        write_json(OUT/'config.json',dict(seed=SEED,max_fp32_steps=a.steps,max_qat_steps=a.qat,
            loss='RGB MSE only',crop=320,batch=2,margin=40,all_sampling_phases=True,native_interval=100,
            all_convolution_parameters_trainable=True,style_loss=False,quantized_colour_levels=False,
            shared_contract_sha256=CONTRACT_SHA,code_sha256=file_hash(__file__),hardware_changed=False))
        shutil.copy2(__file__,OUT/Path(__file__).name)
        preflight(train,OUT/'preflight',a.steps)
    if a.preflight_only:
        print('RECON_READY_FOR_TRAINING',len(train),len(val),flush=True)
        return
    tr = Trainer(train,a.steps)
    train_stage(tr,train,val,OUT/'fp32',min(4000,a.steps))
    best = torch.load(OUT/'fp32/best.pt',map_location='cpu',weights_only=True)
    tr.net.load_state_dict(best['model'])
    tr.net.calibrate()
    tr.net.quantized = True
    state = copy.deepcopy(tr.net.state_dict())
    write_json(OUT/'ptq.json',evaluate(tr.net,val))
    del tr;torch.cuda.empty_cache()
    preflight(train,OUT/'qat-preflight',a.qat,True,state)
    tr = Trainer(train,a.qat,True)
    tr.net.load_state_dict(state)
    train_stage(tr,train,val,OUT/'qat',min(1000,a.qat),interval=250,patience=4)
    best = torch.load(OUT/'qat/best.pt',map_location='cpu',weights_only=True)
    tr.net.load_state_dict(best['model'])
    export_review(tr.net,val,OUT/'package')
    assert file_hash(ROOT/'network_ab/model_b_contract.json') == CONTRACT_SHA
    print('RECONSTRUCTION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
