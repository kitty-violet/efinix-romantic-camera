"""Local causal test: learn actual optimized res3_add codes from real RGB.

The current decoder is frozen. Targets are the 32 signed-byte feature fields
found by audit_b1_current_tail_inverse, not the old 14 ordinal labels. This
three-crop fit is a learning diagnostic, never generalization evidence.
"""
import argparse, copy, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_trainable_luma_readout import ReadoutB1
from .whitened_b1_optimizer import WhitenedB1Adam, patch_statistics
from .train_b1_quantum_regions import RowScaledAdam
from .b1_geometry_first_objective import geometry_first
from .runtime import configure_workspace, write_json, file_hash
from .engine import set_reproducible
from . import train_b1_minimum_area as m


def main():
    ap = argparse.ArgumentParser()
    for key in ('checkpoint', 'data', 'mining', 'latent', 'out'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--interval', type=int, default=200)
    ap.add_argument('--lr', type=float, default=.0002)
    ap.add_argument('--rgb-weight', type=float, default=0.)
    ap.add_argument('--optimizer', choices=['whitened', 'row-adam', 'adam'], default='whitened')
    ap.add_argument('--float-prefix', action='store_true')
    ap.add_argument('--target-input-diagnostic', action='store_true')
    ap.add_argument('--free-tail', action='store_true')
    ap.add_argument('--code-weight', type=float, default=1.)
    a = ap.parse_args()
    configure_workspace(); set_reproducible(2026100801); torch.set_num_threads(2)
    state, initial = load(a.checkpoint)
    initial.__class__ = ReadoutB1; initial.cuda()
    assert not a.float_prefix or a.optimizer == 'adam', 'FP32 diagnostic uses ordinary Adam without quantized DC compensation'
    out = Path(a.out); out.mkdir(exist_ok=False)
    data = m.load_data(a.data); byid = {r['id']: r for r in data['training']}
    mined = json.loads(Path(a.mining).read_text(encoding='utf-8'))['records']
    specs = [mined[i] for i in (1, 5, 6)]
    latent = torch.load(a.latent, map_location='cuda', weights_only=True)
    inverse_report = json.loads(Path(a.latent).with_name('results.json').read_text(encoding='utf-8'))
    assert inverse_report['checkpoint_sha256'] == file_hash(a.checkpoint)
    assert latent['uses_target_at_optimization'] and latent['node'] == 'res3_add'
    assert latent['ids'] == [s['row']['id'] for s in specs]
    xs, ts, ls = [], [], []
    for s in specs:
        row = byid[s['row']['id']]; xx, yy = s['xy']; w, h = s['wh']
        def read(key):
            return torch.from_numpy(m.t.r.read(row[key])[yy:yy+h, xx:xx+w].transpose(2, 0, 1).copy()).float().cuda()/128-1
        xs.append(read('path')); ts.append(read('target'))
        ls.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h, xx:xx+w].astype(np.int64)).cuda())
    x, target, labels = torch.stack(xs), torch.stack(ts), torch.stack(ls)
    if a.target_input_diagnostic: x = target.clone()
    truth = latent['features'].float()
    cut = lambda z: z[..., 40:-40, 40:-40]
    zcut = lambda z: z[..., 10:-10, 10:-10]
    tail_keys = set(); after = False
    for row in initial.plan['nodes']:
        if row['id'] == 'res3_add': after = True; continue
        key = row['id'].replace('.', '__')
        if key in initial.weight:
            initial.weight[key].requires_grad_(not after or a.free_tail)
            initial.bias[key].requires_grad_(not after or a.free_tail)
            if after and not a.free_tail: tail_keys.add(key)
    assert (tail_keys or a.free_tail) and not any(initial.weight[k].requires_grad for k in tail_keys)
    assert not a.float_prefix or not a.free_tail
    assert a.code_weight > 0 or a.rgb_weight > 0
    with torch.no_grad():
        original = initial(x, all_nodes=True)
        baseline = original['rgb_head']/128
        assert original['res3_add'].shape == truth.shape
        ideal = fragment(initial, truth, 'res3_add')['rgb_head']/128
        statistics = patch_statistics(initial, x, 4096)
    initial.quantized = not a.float_prefix
    def optimizer(net):
        groups = []
        for k, w in net.weight.items():
            if not w.requires_grad: continue
            scale = (getattr(net, k+'_m')/131072).float().flatten()
            groups += [dict(params=[w], lr=a.lr, **({} if a.optimizer == 'adam' else dict(row_gain=(scale*256)[:, None, None, None]))),
                       dict(params=[net.bias[k]], lr=.008)]
        return WhitenedB1Adam(groups, net, statistics, True) if a.optimizer == 'whitened' else (RowScaledAdam(groups) if a.optimizer == 'row-adam' else torch.optim.Adam(groups))
    def objective(net):
        nodes = net(x, all_nodes=True)
        # The float diagnostic trains only the prefix; displayed RGB still
        # passes rounded codes through the unchanged true integer decoder.
        p = (fragment(net, net.round(nodes['res3_add']), 'res3_add')['rgb_head'] if a.float_prefix else nodes['rgb_head'])/128
        code = F.smooth_l1_loss(zcut(nodes['res3_add']), zcut(truth), beta=1.)
        rgb = geometry_first(cut(p), cut(target), cut(labels))[0] if a.rgb_weight else code.new_zeros(())
        return a.code_weight*code+a.rgb_weight*rgb, nodes, p
    def update(net, opt):
        opt.zero_grad(); loss, _, _ = objective(net); loss.backward()
        if a.optimizer == 'whitened': opt.prepare_gradients()
        grads = [v for v in net.parameters() if v.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(grads, 5, error_if_nonfinite=True)
        assert norm > 0
        opt.step()
        assert all(torch.equal(net.weight[k], initial.weight[k]) and torch.equal(net.bias[k], initial.bias[k]) for k in tail_keys)
        return float(loss.detach()), float(norm)
    def integer_check(net, nodes):
        with torch.no_grad():
            if a.float_prefix:
                net.quantized = True; nodes = net(x, all_nodes=True); net.quantized = False
            runtime = m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()), device='cuda')
            actual, _ = runtime(net.round(x*128), True)
            errors = {k: float((v-actual[k]).abs().max()) for k, v in nodes.items()}
        assert max(errors.values()) == 0
        return errors
    paths = sorted({Path(mod.__file__).resolve() for name, mod in tuple(sys.modules.items())
                    if name.startswith('network_ab.training') and getattr(mod, '__file__', None)} | {Path(__file__).resolve()})
    hashes = {str(p): file_hash(p) for p in paths}
    for p in paths: shutil.copy2(p, out/p.name)
    write_json(out/'config.json', dict(args=vars(a), checkpoint_sha256=file_hash(a.checkpoint),
        latent_sha256=file_hash(a.latent), data_sha256=file_hash(a.data), specs=specs,
        frozen_decoder=sorted(tail_keys), code=hashes, local_fit_not_generalization=True,
        actual_source_inference=not a.target_input_diagnostic,
        precision='FP32_noAMP_noTF32', extra_inference_operations=0))
    # Actual GPU minibatch, disk reload and bit-exact next update, then reset.
    net = copy.deepcopy(initial); opt = optimizer(net)
    first = update(net, opt)
    torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict()), out/'preflight.pt')
    expected = update(net, opt)
    other = copy.deepcopy(initial); op = optimizer(other)
    disk = torch.load(out/'preflight.pt', map_location='cuda', weights_only=True)
    other.load_state_dict(disk['model']); op.load_state_dict(disk['optimizer'])
    got = update(other, op)
    err = max(float((v-other.state_dict()[k]).abs().max()) for k, v in net.state_dict().items())
    assert err == 0 and expected == got
    with torch.no_grad(): _, nodes, _ = objective(net)
    integer_check(net, nodes)
    write_json(out/'preflight.json', dict(status='PASS', first=first, next_update=expected,
        actual_disk_resume_error=err, actual_all_nodes_error=0, frozen_tail_unchanged=True))
    print('COMPATIBLE_CODES_PREFLIGHT_PASS GPU/backward/minibatch/disk_resume0/all_nodes0', flush=True)
    del net, opt, other, op, disk, nodes; torch.cuda.empty_cache()
    net = copy.deepcopy(initial); opt = optimizer(net); history = []; started = time.monotonic()
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 13)
    for step in range(a.steps+1):
        if step % a.interval == 0 or step == a.steps:
            assert all(file_hash(p) == v for p, v in hashes.items()), 'Executing source changed'
            with torch.no_grad():
                loss, nodes, p = objective(net)
                record = dict(step=step, loss=float(loss), code_mae=float((zcut(nodes['res3_add'])-zcut(truth)).abs().mean()),
                    rgb_mae=float((cut(p)-cut(target)).abs().mean()*128),
                    ideal_rgb_mae=float((cut(ideal)-cut(target)).abs().mean()*128))
                if a.float_prefix:
                    net.quantized = True; actual = net(x, all_nodes=True); net.quantized = False
                    record['quantized_source_rgb_mae'] = float((cut(actual['rgb_head']/128)-cut(target)).abs().mean()*128)
            integer_check(net, nodes)
            board = Image.new('RGB', (1280, 1044), '#17202a'); draw = ImageDraw.Draw(board)
            for i in range(3):
                for j, (title, values) in enumerate((('固定目标', target), ('初始目标输入·仅诊断' if a.target_input_diagnostic else '初始源预测', baseline), ('目标输入·仅诊断' if a.target_input_diagnostic else '浮点前段·仅诊断' if a.float_prefix else '实际源预测·仅三裁片训练', p), ('已知目标逆解·仅诊断', ideal))):
                    board.paste(m.t.r.c.rgb_image(values[i]), (320*j, 348*i+28))
                    draw.text((320*j+3, 348*i+3), title, font=font, fill='white')
            board.save(out/f'local_{step:06d}.png')
            saved = copy.deepcopy(state)
            saved.update(model=net.state_dict(), optimizer=opt.state_dict(), step=step,
                         decoder_compatible_code_fit=True, local_fit_only=True, visual_acceptance=False,
                         inverse_target_sha256=file_hash(a.latent), frozen_decoder=sorted(tail_keys),
                         prefix_float_diagnostic=a.float_prefix, optimizer_kind=a.optimizer,
                         free_tail=a.free_tail, code_weight=a.code_weight)
            saved['target_input_diagnostic'] = a.target_input_diagnostic
            torch.save(saved, out/f'local_{step:06d}.pt')
            history.append(record); write_json(out/'history.json', history)
            print('COMPATIBLE_CODES_LOCAL', record, flush=True)
        if step == a.steps: break
        update(net, opt)
    write_json(out/'review.json', dict(history=history, seconds=time.monotonic()-started,
        all_nodes_error=0, frozen_decoder_unchanged=True, actual_source_inference=not a.target_input_diagnostic,
        local_fit_not_generalization=True, visual_acceptance=False))


if __name__ == '__main__': main()
