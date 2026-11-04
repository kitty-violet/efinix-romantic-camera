"""Suppress local bright excursions without a whole-region darkening gradient.

Only the training objective changes. The original fixed B1 integer graph and
the packed720 coefficient representation remain unchanged.
"""
from pathlib import Path
import torch
from .train_b1_ternary_packed720 import (
    Packed720Training, bright_envelope_loss, evaluate, preview, driver,
)


def centred_region_luma(p, t, labels):
    yp = .25*p[:, 0] + .5*p[:, 1] + .25*p[:, 2]
    yt = .25*t[:, 0] + .5*t[:, 1] + .25*t[:, 2]
    centred = []
    for pp, tt, ll in zip(yp, yt, labels):
        _, index, count = torch.unique(ll.flatten(), return_inverse=True,
                                       return_counts=True)
        n = len(count)
        mp = pp.new_zeros(n).scatter_add(0, index, pp.flatten()) / count
        mt = tt.new_zeros(n).scatter_add(0, index, tt.flatten()) / count
        centred.append(pp - (mp[index] - mt[index]).reshape_as(pp))
    return torch.stack(centred)[:, None], yt[:, None]


def local_rim_loss(p, t, labels):
    centred, target = centred_region_luma(p, t, labels)
    return bright_envelope_loss(centred.expand(-1, 3, -1, -1),
                                target.expand(-1, 3, -1, -1))


class LocalRimTraining(Packed720Training):
    def additional_loss(self, p, t, native):
        loss, values = super().additional_loss(p, t, native)
        labels = self.labels if native else self.labels[..., 40:-40, 40:-40]
        local = local_rim_loss(p, t, labels)
        loss = loss - 4*values['bright_envelope'] + 4*local
        values['mean_centred_local_bright_rim'] = local
        return loss, values

    def save(self, path):
        super().save(path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        state.update(bright_rim_objective='differentiable_region_mean_centred',
                     bright_rim_region_dc_gradient_zero=True)
        torch.save(state, path)


def regression():
    labels = torch.zeros(1, 64, 96, dtype=torch.long)
    labels[..., 48:] = 1
    t = torch.zeros(1, 3, 64, 96)
    t[..., 48:] = .6
    assert local_rim_loss(t, t, labels).abs() < 1e-7
    # A legitimate nearest-palette region colour can be above or below the
    # target. This loss must not choose its global brightness.
    offset = t.clone()
    offset[..., :48] += 10/128
    offset[..., 48:] -= 10/128
    assert local_rim_loss(offset, t, labels).abs() < 1e-7
    p = offset.clone()
    p[..., 24:28, 24:28] += .3
    p.requires_grad_()
    loss = local_rim_loss(p, t, labels)
    loss.backward()
    assert loss > .02 and torch.isfinite(p.grad).all()
    assert p.grad[..., 24:28, 24:28].sum() > 0
    for region in (0, 1):
        gradient_sum = (p.grad * (labels == region)[:, None]).sum((2, 3))
        assert gradient_sum.abs().max() < 1e-6, gradient_sum
    assert p.grad[..., 10:20, 10:20].sum() < 0
    print('LOCAL_RIM_REGRESSION_PASS', dict(local_bright_penalty=float(loss.detach()),
          real_white_preserved=True, region_offset_neutral=True,
          gradient_region_sum_zero=True), flush=True)


if __name__ == '__main__':
    import sys
    torch.set_num_threads(2)
    regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=LocalRimTraining,
                    extra_sources=[Path(__file__)], evaluation=evaluate,
                    preview=preview)
