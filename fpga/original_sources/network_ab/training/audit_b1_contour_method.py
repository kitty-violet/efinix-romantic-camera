"""Read-only diagnosis of the contour-to-colour assumption, not a new model.

Counterfactual gates below are explicitly diagnostic and are never exported.
Actual output is regenerated from the existing serialized B1 coefficient image.
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1, half
from .train_b1_deep_contour import OUT as LABELS
from .train_b1_late_contour import OUT as RUN, LateStudent
from .train_b1_structural_gate import CROPS

OUT = RUN.parent / 'b1-contour-method-audit-20261005-v1'


@torch.inference_mode()
def replay(net, x, replacement=None):
    values = {'input': x}
    for row in net.plan['nodes']:
        name = row['id']
        a = values[row['inputs'][0]]
        if name in net.ops:
            w, b, m = net.ops[name]
            if row['kernel'] == 3:
                a = F.pad(a, (1, 1, 1, 1), mode='replicate')
            dot = F.conv2d(a, w, stride=row['stride'], groups=row['groups'])
            y = half((dot.double()+b).clamp(-(2**31), 2**31-1)*m/131072).clamp(-128, 127).float()
            if row['activation'] == 'relu':
                y = y.clamp_min(0)
        elif row['kind'] == 'nearest_x2':
            y = a.repeat_interleave(2, -2).repeat_interleave(2, -1)
        else:
            z = values[row['inputs'][1]]
            r = net.package['residual_encoding'][name]
            small, other = (a, z) if r['main_scaled'] else (z, a)
            y = half((half(small.double()*r['multiplier']/131072).clamp(-128, 127)
                      + other.double())*r['post_multiplier']/131072).clamp(-128, 127).float()
        if name == 'res3_add' and replacement is not None:
            y = y.clone()
            y[:, 24:26] = replacement.expand(-1, 2, -1, -1)
        values[name] = y
    return values


def as_image(nodes):
    return Image.fromarray((nodes['rgb_head'][0].permute(1, 2, 0).cpu().numpy()+128).astype(np.uint8))


def allocation():
    net = LateStudent()
    per_layer = {}
    for kind, params in [('w', net.weight), ('b', net.bias)]:
        for key, value in params.items():
            mask = getattr(net, kind+'_'+key+'_mask', None)
            count = int(mask.sum()) if value.requires_grad and mask is not None else 0
            per_layer.setdefault(key, {})[kind] = dict(total=value.numel(), enabled=count)
    # This is enabled parameter count, not a claim about effective capacity.
    return dict(layers=per_layer,
                enabled=sum(v[k]['enabled'] for v in per_layer.values() for k in ('w', 'b')),
                total=sum(v[k]['total'] for v in per_layer.values() for k in ('w', 'b')),
                decoder_trainable=any(p.requires_grad for k, p in net.weight.items()
                                      if k.startswith('up') or k == 'rgb_head'))


@torch.inference_mode()
def main():
    configure_workspace()
    set_reproducible(2026100516)
    contract_guard()
    OUT.mkdir(exist_ok=False)
    source = RUN / 'step_002200/coefficients-padded.bin'
    initial = RUN / 'initial_integer/coefficients-padded.bin'
    net = IntegerB1(runtime_from_image(source.read_bytes()))
    old = IntegerB1(runtime_from_image(initial.read_bytes()))
    data = json.loads((LABELS/'data.json').read_text(encoding='utf-8'))
    records, panels = [], {}
    for r in data:
        if r['id'] not in {c[0] for c in CROPS}:
            continue
        rgb = np.array(Image.open(r['source']).convert('RGB'))
        x = torch.tensor(rgb.transpose(2, 0, 1)[None].copy(), device='cuda', dtype=torch.float32)-128
        nodes, _ = net(x, True)
        repeated = replay(net, x)
        replay_error = max(float((v-repeated[k]).abs().max()) for k, v in nodes.items())
        assert replay_error == 0
        del repeated
        before, _ = old(x, True)
        carrier_error = float((nodes['res3_add'][:, :6]-before['res3_add'][:, :6]).abs().max())
        assert carrier_error == 0
        del before
        a = nodes['up1_low_dw'][:, :3]
        mu = nodes['up1_low_dw'][:, 3:6]
        e = nodes['up1_low_dw'][:, 24:25]
        u = nodes['up1_low_pw']
        merged = u[:, :3]-u[:, 3:6]+u[:, 6:9]
        formula = mu+torch.maximum(torch.minimum(a-mu, e), -e)
        formula_error = float((merged-formula).abs().max())
        assert formula_error == 0
        target = half(torch.tensor(np.load(r['label']).astype(np.float32)[::4, ::4],
                                  device='cuda')[None, None]*32-8)
        variants = {'zero_gate': torch.zeros_like(target), 'target_gate': target,
                    'full_preserve': torch.full_like(target, 127)}
        panels[r['id']] = dict(teacher=Image.open(r['target']).convert('RGB'), actual=as_image(nodes))
        row = dict(id=r['id'], replay_error=replay_error, frozen_carrier_error=carrier_error,
                   exact_integer_merge_formula_error=formula_error, variants={})
        valid = torch.zeros_like(e, dtype=torch.bool)
        valid[:, :, 8:-8, 8:-8] = True
        empty = (target <= 0) & valid
        row['weak_false_positive_fraction_on_empty'] = float(((e > 0) & empty).sum()/empty.sum())
        row['strong_false_positive_fraction_on_empty'] = float(((e > 4) & empty).sum()/empty.sum())
        for name, value in variants.items():
            changed = replay(net, x, value)
            delta = changed['rgb_head']-nodes['rgb_head']
            row['variants'][name] = dict(rgb_mae_to_actual=float(delta.abs().mean()),
                                        changed_pixel_fraction=float((delta != 0).any(1).float().mean()))
            panels[r['id']][name] = as_image(changed)
            del changed
        records.append(row)
        print('METHOD_AUDIT', r['id'], json.dumps(row, ensure_ascii=False), flush=True)
        del nodes
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    columns = [('teacher', '冻结彩色目标'), ('actual', '现有真实权重输出'),
               ('target_gate', '仅诊断：强制替换为标签轮廓'),
               ('zero_gate', '仅诊断：全部取局部均色'), ('full_preserve', '仅诊断：全部保留载体颜色')]
    board = Image.new('RGB', (420*len(columns), 314*len(CROPS)), '#17202a')
    draw = ImageDraw.Draw(board)
    for i, (key, box, label) in enumerate(CROPS):
        for j, (kind, title) in enumerate(columns):
            board.paste(panels[key][kind].crop(box), (420*j, 314*i+34))
            draw.text((420*j+4, 314*i+4), title, font=font, fill='white')
    board.save(OUT/'native_counterfactuals.png')
    # Compact original-scale crop, avoiding a large board being downscaled by UI.
    board.crop((0, 0, 1260, 314)).save(OUT/'grass_actual_vs_target_gate.png')
    write_json(OUT/'evidence.json', dict(
        diagnostic_only=True, new_training=False, new_weights=False, actual_board=False,
        source=str(source), source_sha256=file_hash(source), initial_sha256=file_hash(initial),
        allocation=allocation(), records=records,
        formula='merged = mean + clip(carrier - mean, -gate, +gate), before fixed colour quantization',
        caveat='This diagnoses this frozen decoder, not impossibility for other B1 weight allocations.'))
    shutil.copy2(__file__, OUT/Path(__file__).name)
    print('METHOD_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
