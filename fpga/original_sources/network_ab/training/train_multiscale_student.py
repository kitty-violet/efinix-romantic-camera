"""Isolated bounded fitting and source-separated FP32 multiscale distillation."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .multiscale_student import MultiscaleStudent, profile
from .train_b1_distillation import as_tensor, to_rgb, read_image, read_flow, edges
from .coherent_shared_model import warp
from .losses import PerceptualLoss, gram
from .engine import set_reproducible
from .runtime import configure_workspace, file_hash, write_json, contract_guard

RUN = 'multiscale-student-20261003'


def objective(pred, source, target, vgg, margin, flow=None, mask=None):
    cut = lambda a: a[:, :, margin:-margin, margin:-margin]
    p, x, t = [cut(a) for a in (pred, source, target)]
    focus = 1 + ((t-x).abs().mean(1, keepdim=True)/.08).clamp(0, 3)
    rgb = ((p-t).abs()*focus).sum()/(3*focus.sum())
    # Direct targets at several scales, not magnification of 9px high frequency.
    planes = sum(F.l1_loss(F.avg_pool2d(p, k), F.avg_pool2d(t, k)) for k in (4, 16, 32))/3
    edge = sum(F.l1_loss(a, b) for a, b in zip(edges(p), edges(t)))
    with torch.no_grad():
        tf = vgg.extract(t)
    pf = vgg.extract(p)
    perceptual = F.l1_loss(pf['relu2_2'], tf['relu2_2'])
    style = F.l1_loss(gram(pf['relu2_2']), gram(tf['relu2_2']))
    terms = dict(rgb=rgb, planes=planes, edge=edge, perceptual=perceptual, gram=style)
    if flow is not None:
        n = len(pred)//2
        delta = (pred[n:]-source[n:])-warp(pred[:n]-source[:n], flow)
        valid = cut(mask)
        terms['temporal'] = (cut(delta).abs()*valid).sum()/(3*valid.sum()+1e-6)
    return terms


class Training:
    def batch(self, native=False):
        values=[]; flows=[]; masks=[]; specs=[]
        for _ in range(1 if native else self.config['pairs_per_step']):
            row=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))])
            if native: x,y,w,h=0,0,1280,720
            else:
                w=h=self.config['crop']
                x=8*self.rng.randrange((1280-w)//8+1); y=8*self.rng.randrange((720-h)//8+1)
            values.extend([as_tensor(read_image(row[k])[y:y+h,x:x+w]) for k in ('current','next','current_target','next_target')])
            flow,mask=read_flow(row['flow']); f=flow[y:y+h,x:x+w].copy(); m=mask[y:y+h,x:x+w].copy()
            yy,xx=np.mgrid[:h,:w]; margin=self.config['margin']
            m*=((xx+f[:,:,0]>=margin)&(xx+f[:,:,0]<w-margin)&(yy+f[:,:,1]>=margin)&(yy+f[:,:,1]<h-margin))
            flows.append(torch.from_numpy(f).permute(2,0,1)); masks.append(torch.from_numpy(m)[None])
            specs.append(dict(source=row['source'],index=row['current_index'],xy=[x,y],wh=[w,h]))
        n=len(specs);v=torch.stack(values).cuda().reshape(n,4,3,h,w)
        return v[:,0],v[:,1],v[:,2],v[:,3],torch.stack(flows).cuda(),torch.stack(masks).cuda(),specs

    def __init__(self, root, cfg, manifest, size, warm=None):
        self.step = 0
        self.config = cfg
        self.manifest = manifest
        self.groups = {}
        self.rng = random.Random(cfg['seed'])
        for row in manifest['pairs']:
            assert manifest['source_splits'][row['source']]['split'] == 'train'
            self.groups.setdefault(row['source'], []).append(row)
        torch.manual_seed(cfg['seed'])
        self.net = MultiscaleStudent(size).cuda().eval()
        if warm:
            self.net.load_state_dict(torch.load(warm, weights_only=True, map_location='cpu')['model'])
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg['lr'])
        self.vgg = PerceptualLoss(root).cuda()
        self.vgg.features = self.vgg.features[:9]

    def update(self, native=False, fixed=None):
        start = time.perf_counter()
        self.net.eval()
        if fixed is None:
            a, b, ta, tb, flow, mask, specs = self.batch(native)
            x, t = torch.cat([a, b]), torch.cat([ta, tb])
        else:
            x, t = fixed
            flow = mask = None
            specs = []
        p = self.net(x)
        terms = objective(p, x, t, self.vgg, self.config['margin'], flow, mask)
        loss = sum(v*self.config['weights'][k] for k, v in terms.items())
        progress = min(1., self.step/max(self.config['max_steps']-1, 1))
        lr = (self.config['lr_floor']+(self.config['lr']-self.config['lr_floor'])*.5*(1+math.cos(math.pi*progress)))*min(1., (self.step+1)/20)
        for group in self.opt.param_groups:
            group['lr'] = lr
        self.opt.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
        loss.backward()
        for name, param in self.net.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5., error_if_nonfinite=True)
        assert norm > 0
        self.opt.step()
        self.step += 1
        torch.cuda.synchronize()
        return dict(step=self.step, loss=float(loss.detach()), grad_norm=float(norm), lr=lr,
                    **{k: float(v.detach()) for k, v in terms.items()}, samples=specs, native=native,
                    seconds=time.perf_counter()-start, peak_mib=torch.cuda.max_memory_allocated()/2**20)

    def state(self, binding):
        return dict(model=self.net.state_dict(), optimizer=self.opt.state_dict(), step=self.step,
                    sampler_rng=self.rng.getstate(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), binding=binding, config=self.config, size=self.net.size)

    def restore(self, state, binding):
        assert state['binding'] == binding and state['config'] == self.config and state['size'] == self.net.size
        self.net.load_state_dict(state['model'])
        self.opt.load_state_dict(state['optimizer'])
        self.rng.setstate(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.step = state['step']


def prepare(root, run):
    run.mkdir(exist_ok=False)
    original = root/'runs/fullres-c8-stylepush-color-20261003/data_manifest.json'
    manifest = json.loads(original.read_text(encoding='utf-8'))
    for row in manifest['pairs']:
        for key in ('current', 'next', 'current_target', 'next_target', 'flow'):
            assert file_hash(row[key]) == row[key+'_sha256']
    for row in manifest['validation']:
        for key in ('source', 'target'):
            assert file_hash(row[key]) == row[key+'_sha256']
    shutil.copy2(original, run/'data_manifest.json')
    cfg = dict(seed=202610038, max_steps=2000, lr=.0015, lr_floor=.00015, pairs_per_step=1,
               crop=512, margin=96, native_interval=50, weights=dict(rgb=1., planes=.5, edge=2., perceptual=.12, gram=.015, temporal=.2),
               fp32=True, amp=False, tf32=False, ema=False, target='Original corrected teacher, no target exaggeration',
               sizes=['small', 'medium'], fit_steps=600, fit_lr=.002)
    groups = {}
    for row in manifest['pairs']:
        groups.setdefault(row['source'], []).append(row)
    selected = []
    for i, xy in [(2, (512, 144)), (3, (384, 160)), (5, (448, 200))]:
        rows = groups[sorted(groups)[i]]
        row = rows[len(rows)//2]
        selected.append(dict(id=Path(row['current']).stem, source=row['current'], target=row['current_target'],
                             source_sha256=row['current_sha256'], target_sha256=row['current_target_sha256'],
                             raw_source=row['source'], xy=list(xy), crop=512))
    cfg['fit_regions'] = selected
    write_json(run/'config.json', cfg)
    paths = [Path(__file__), Path(__file__).with_name('multiscale_student.py'), Path(__file__).with_name('train_b1_distillation.py'),
             Path(__file__).with_name('coherent_shared_model.py'), Path(__file__).with_name('losses.py'),
             Path(__file__).with_name('runtime.py'), Path(__file__).with_name('engine.py')]
    source = run/'source'; source.mkdir()
    for path in paths:
        shutil.copy2(path, source/path.name)
    binding = dict(config_sha256=file_hash(run/'config.json'), manifest_sha256=file_hash(run/'data_manifest.json'),
                   code={str(p):file_hash(p) for p in paths}, contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'))
    write_json(run/'binding.json', binding)
    assert {r['raw_source'] for r in selected}.isdisjoint({r['raw_source'] for r in manifest['validation']})
    write_json(run/'data_check.json', dict(status='PASS', train_sources=7, train_pairs=95, source_isolation=True,
               fit_uses_training_sources_only=True, regression_previously_reviewed_not_blind=True))
    print('PREPARED', run, flush=True)


def fixed_inputs(cfg):
    xs, ts = [], []
    for row in cfg['fit_regions']:
        x, y = row['xy']; c = row['crop']
        xs.append(as_tensor(read_image(row['source'])[y:y+c, x:x+c]))
        ts.append(as_tensor(read_image(row['target'])[y:y+c, x:x+c]))
    return torch.stack(xs).cuda(), torch.stack(ts).cuda()


def evaluate(trainer, cfg, manifest, directory):
    directory.mkdir(exist_ok=False)
    from .b1_fit_controls import score
    records = []
    with torch.inference_mode():
        for row in cfg['fit_regions'] + manifest['validation']:
            key = row['id']; source = read_image(row['source']); target = read_image(row['target'])
            p = to_rgb(trainer.net(as_tensor(source)[None].cuda())[0])
            Image.fromarray(p).save(directory/f'{key}.png')
            box = (*row['xy'],) if 'xy' in row else ((424, 320) if 'stairs' in key else (80, 400) if 'plaza' in key else (560, 150))
            x, y = box
            if 'xy' in row:
                x += cfg['margin']; y += cfg['margin']
            a, t, z = [v[y:y+320, x:x+320] for v in (source, target, p)]
            panel = Image.new('RGB', (960, 348), '#181d25'); draw = ImageDraw.Draw(panel)
            for j, (im, label) in enumerate([(a, 'Source'), (t, 'Teacher'), (z, trainer.net.size+' / '+str(trainer.step))]):
                panel.paste(Image.fromarray(im), (320*j, 28)); draw.text((320*j+8, 8), label, fill='white')
            panel.save(directory/f'{key}_comparison.png')
            records.append(dict(id=key, role='fit_training' if 'xy' in row else 'reviewed_regression',
                                teacher=score(z, t), source=score(z, a)))
    write_json(directory/'scores.json', dict(records=records))


def preflight(root, run, cfg, manifest, binding):
    out = run/'preflight'; out.mkdir(exist_ok=False)
    results = []
    for size in cfg['sizes']:
        trainer = Training(root, cfg, manifest, size)
        x = as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda()
        info = profile(trainer.net, x)
        full = trainer.update(True)
        crop = trainer.update(False)
        p = out/(size+'_resume.pt'); torch.save(trainer.state(binding), p)
        reference = trainer.update(False)
        expected = {k:v.detach().cpu().clone() for k,v in trainer.net.state_dict().items()}
        del trainer; gc.collect(); torch.cuda.empty_cache()
        trainer = Training(root, cfg, manifest, size)
        trainer.restore(torch.load(p, weights_only=True, map_location='cpu'), binding)
        replay = trainer.update(False)
        error = max(float((v.cpu()-expected[k]).abs().max()) for k,v in trainer.net.state_dict().items())
        assert error < 3e-6 and reference['samples'] == replay['samples'], error
        assert all(v.dtype == torch.float32 for v in trainer.net.parameters())
        results.append(dict(size=size, profile=info, full_backward=full, crop_backward=crop, restore_error=error))
        del trainer; gc.collect(); torch.cuda.empty_cache()
    write_json(out/'result.json', dict(status='PASS', binding=binding, records=results))
    print('PREFLIGHT_PASS', json.dumps([{k:r[k] for k in ('size','restore_error')} for r in results]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'preflight', 'fit', 'train'])
    ap.add_argument('--size', choices=['small', 'medium'], default='small')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--stop-after', type=int)
    args = ap.parse_args()
    root = configure_workspace(); run = root/'runs'/RUN
    contract_guard(); set_reproducible(202610038); cv2.setNumThreads(2)
    if args.mode == 'prepare':
        prepare(root, run); return
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    binding = json.loads((run/'binding.json').read_text(encoding='utf-8'))
    assert file_hash(run/'config.json') == binding['config_sha256']
    assert file_hash(run/'data_manifest.json') == binding['manifest_sha256']
    for p, digest in binding['code'].items():
        assert file_hash(p) == digest, p
    if args.mode == 'preflight':
        preflight(root, run, cfg, manifest, binding); return
    pre = json.loads((run/'preflight/result.json').read_text()); assert pre['status']=='PASS' and pre['binding']==binding
    fitting = args.mode == 'fit'
    cfg = copy.deepcopy(cfg)
    if fitting:
        cfg.update(max_steps=cfg['fit_steps'], lr=cfg['fit_lr'], lr_floor=.0002)
    out = run/(('fit_' if fitting else 'train_')+args.size)
    if not args.resume:
        out.mkdir(exist_ok=False)
    warm = None if fitting else run/('fit_'+args.size)/'weights_fp32.pt'
    if not fitting:
        acceptance = json.loads((run/'fit_review.json').read_text(encoding='utf-8'))
        assert acceptance['selected']==args.size and acceptance['continue_multisource']
        assert file_hash(warm)==acceptance['warmstart_sha256']
    trainer = Training(root, cfg, manifest, args.size, warm)
    if args.resume:
        trainer.restore(torch.load(args.resume, weights_only=True, map_location='cpu'), binding)
        assert [json.loads(s)['step'] for s in (out/'metrics.jsonl').read_text().splitlines()]==list(range(1, trainer.step+1))
    else:
        evaluate(trainer, cfg, manifest, out/'validation_000000')
    write_json(out/f'launch_{trainer.step:06d}.json', dict(step=trainer.step, resume=str(args.resume), warmstart=str(warm), binding=binding))
    fixed = fixed_inputs(cfg) if fitting else None
    stop = min(args.stop_after or cfg['max_steps'], cfg['max_steps'])
    with (out/'metrics.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        while trainer.step < stop:
            row = trainer.update(not fitting and (trainer.step+1)%cfg['native_interval']==0, fixed)
            log.write(json.dumps(row, ensure_ascii=False)+'\n')
            if trainer.step%25==0:
                write_json(out/'progress.json', row)
                print(args.mode, args.size, trainer.step, round(row['loss'],5), round(row['seconds'],3), flush=True)
            if trainer.step%200==0 or trainer.step==stop:
                torch.save(trainer.state(binding), out/f'checkpoint_{trainer.step:06d}.pt')
                evaluate(trainer, cfg, manifest, out/f'validation_{trainer.step:06d}')
    complete = trainer.step == cfg['max_steps']
    write_json(out/'status.json', dict(status='COMPLETE' if complete else 'BOUNDED_SEGMENT_COMPLETE', step=trainer.step))
    if complete:
        path = out/'weights_fp32.pt'
        torch.save(dict(schema='multiscale-student-fp32-v1', size=args.size, model={k:v.cpu() for k,v in trainer.net.state_dict().items()},
                        step=trainer.step, fitting_only=fitting, binding=binding, production=False), path)
        reload = MultiscaleStudent(args.size).cuda().eval()
        reload.load_state_dict(torch.load(path, weights_only=True)['model'])
        x = as_tensor(read_image(manifest['validation'][0]['source']))[None].cuda()
        with torch.inference_mode(): error = float((reload(x)-trainer.net(x)).abs().max())
        assert error == 0
        write_json(out/'export.json', dict(path=str(path), sha256=file_hash(path), reload_error=error, step=trainer.step, fitting_only=fitting))
    print('SEGMENT_COMPLETE', args.mode, args.size, trainer.step, flush=True)


if __name__ == '__main__':
    main()
