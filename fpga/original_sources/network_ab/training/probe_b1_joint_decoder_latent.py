"""TRAIN-only exact-integer decoder inversion, never source prediction.

Freeze every ordinary B1 coefficient; optimize only the injected 180p feature
map for three known target crops. This localizes encoder/decoder limitations.
It neither produces deployable weights nor proves a global representational
limit. Compare the existing and narrow threshold at the same crop/init/loss.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont

from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .train_b1_aligned_decoder_regions import decoder_logits
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_joint_region_colour import joint_region_loss
from .probe_b1_native_boundary_phase import install
from .runtime import configure_workspace, contract_guard, write_json, file_hash
from .engine import set_reproducible


IDS = ('a118bccbfa7a_000420_source', 'b4fbd43df8cc_000181_source',
       '569862ec7600_000272_source')
BOX = (448, 168, 832, 552)  # Same central TRAIN crop, aligned to four pixels.
MARGIN = 24


def decode(net, z, gradients):
    first = fragment(net, net.round(z).clamp(0, 127), 'up1_low_pw')
    if not gradients:
        return first['rgb_head'] / 128
    proxy = F.pad(127 * decoder_logits(net, first).sigmoid(), (0, 0, 0, 0, 0, 1))
    field = ExactValueProxyGradient.apply(first['up2_low_dw'], proxy)
    return fragment(net, field, 'up2_low_dw')['rgb_head'] / 128


def inner(x):
    return x[..., MARGIN:-MARGIN, MARGIN:-MARGIN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--name', default='joint-decoder-latent-diagnostic')
    a = ap.parse_args()
    configure_workspace(); contract_guard(); set_reproducible(20261006249)
    dest = m.OUT / a.name; dest.mkdir(exist_ok=False)
    data = m.load_data(m.OUT / 'hierarchical-targets-v2/data.json')
    by = {r['id']: r for r in data['training']}
    assert all(key in by for key in IDS)
    rows = [by[key] for key in IDS]
    x0, y0, x1, y1 = BOX
    sources = [m.t.r.read(r['path'])[y0:y1, x0:x1] for r in rows]
    targets = [m.t.r.read(r['target'])[y0:y1, x0:x1] for r in rows]
    x = torch.stack([m.t.r.tensor(im) for im in sources])
    target = torch.stack([m.t.r.tensor(im) for im in targets])
    labels = torch.tensor(np.stack([m.load_region(r['region'])[y0:y1, x0:x1]
                                  for r in rows]).astype(np.int64), device=x.device)
    checkpoint = m.OUT / 'joint-field-colour-qat/candidate_000500.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net = m.t.r.c.ColourB1(revive=False, source=m.t.r.OUT/'initial_identity.bin').cuda()
    net.load_state_dict(state['model']); net.package['residual_encoding'] = copy.deepcopy(state['residual_encoding'])
    net.quantized = True; net.requires_grad_(False)
    initial = copy.deepcopy(net.state_dict()); pictures = {}; results = []
    for name, span in (('current', 5), ('narrow', 1)):
        net.load_state_dict(initial)
        if name != 'current': install(net, span, 'halfgauss')
        raw = net.export()
        with torch.no_grad():
            full = net(x, all_nodes=True)
            z0 = full['up1_low_pw'].clone()
            part = fragment(net, z0, 'up1_low_pw')
            errors = {k: float((v-full[k]).abs().max()) for k, v in part.items()
                      if k not in ('res1_add', 'res2_add')}
            assert max(errors.values()) == 0
            # Independently check all ordinary source nodes against actual bytes.
            actual = m.t.r.IntegerB1(m.t.r.runtime_from_image(raw), device='cuda')
            replay, _ = actual(x*128, True)
            assert max(float((v-replay[k]).abs().max()) for k, v in full.items() if k != 'input') == 0
            base = decode(net, z0, False)
            baseline_loss, baseline_values = joint_region_loss(inner(base), inner(target), inner(labels))
        z = z0.detach().clone().requires_grad_(True)
        assert torch.equal(decode(net, z, True).detach(), base)
        opt = torch.optim.Adam([z], lr=.35)
        history = []; start = time.perf_counter()
        for step in range(1, a.steps+1):
            p = decode(net, z, True)
            loss, values = joint_region_loss(inner(p), inner(target), inner(labels))
            opt.zero_grad(set_to_none=True); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
            opt.step()
            with torch.no_grad(): z.clamp_(0, 127)
            if step % 100 == 0 or step == a.steps:
                with torch.no_grad():
                    p = decode(net, z, False)
                    value, terms = joint_region_loss(inner(p), inner(target), inner(labels))
                row = dict(step=step, loss=float(value), **{k: float(v) for k, v in terms.items()})
                history.append(row); print('LATENT_DIAGNOSTIC', name, row, flush=True)
        with torch.no_grad(): fitted = decode(net, z, False)
        assert net.export() == raw, 'No model coefficients may change in this diagnostic'
        folder = dest/name; folder.mkdir()
        torch.save(dict(injected_integer_feature=z.detach().round().clamp(0,127).cpu(),
                        train_ids=list(IDS), box=BOX, source_prediction=False,
                        deployable=False), folder/'diagnostic_features.pt')
        pictures[name] = (base.detach().cpu(), fitted.detach().cpu())
        records = []
        for i, row in enumerate(rows):
            records.append(dict(id=row['id'], source=m.t.r.metrics(inner(base[i:i+1]), inner(target[i:i+1])),
                                injected=m.t.r.metrics(inner(fitted[i:i+1]), inner(target[i:i+1]))))
            m.t.r.c.rgb_image(fitted[i]).save(folder/(row['id']+'_injected.png'))
        results.append(dict(variant=name, source_node_error=0, fragment_errors=errors,
                            ordinary_weights_unchanged=True, seconds=time.perf_counter()-start,
                            initial_loss=float(baseline_loss), initial_terms={k:float(v) for k,v in baseline_values.items()},
                            history=history, records=records))
        write_json(dest/'results.json', dict(results=results, code_sha256=file_hash(__file__),
            checkpoint_sha256=file_hash(checkpoint), data_manifest_sha256=file_hash(m.OUT/'hierarchical-targets-v2/data.json'),
            training_only=True, source_prediction=False, deployable=False,
            global_optimum_proven=False, temporal_acceptance=False, train_ids=list(IDS), box=BOX))
    size = 384 - 2*MARGIN
    board = Image.new('RGB', (size*5, (size+40)*len(rows)), '#17202a')
    draw = ImageDraw.Draw(board); font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    titles = ['训练原图', '修订目标', '真实网络预测', '注入优化·当前解码器', '注入优化·窄判定']
    for i in range(len(rows)):
        imgs = [inner(x[i]), inner(target[i]), inner(pictures['current'][0][i]),
                inner(pictures['current'][1][i]), inner(pictures['narrow'][1][i])]
        for j, (im, title) in enumerate(zip(imgs, titles)):
            board.paste(m.t.r.c.rgb_image(im), (size*j, (size+40)*i+40))
            draw.text((size*j+5, (size+40)*i+5), title, font=font, fill='white')
    board.save(dest/'comparison.png')
    print('LATENT_DIAGNOSTIC_COMPLETE_NOT_DEPLOYABLE', str(dest), flush=True)


if __name__ == '__main__': main()
