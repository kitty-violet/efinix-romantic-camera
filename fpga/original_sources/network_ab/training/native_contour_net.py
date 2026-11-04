"""Contour-only diagnostic: native-pixel localization + multiscale context.

Central/angular pixel differences follow the algebra described by PiDiNet
(https://github.com/hellozhuo/pidinet/blob/master/models/ops.py). They are
reparameterizations of ordinary 3x3 kernels; exported inference has no extra
difference operator. This is a new graph, NOT a board-validated B1 replacement.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class DifferenceConv(nn.Conv2d):
    def __init__(self, ci, co, stride=1, groups=1, difference=True):
        super().__init__(ci, co, 3, stride, 0, groups=groups)
        self.difference = difference

    def effective_weight(self):
        if not self.difference:
            return self.weight
        w = self.weight.flatten(2)
        center = torch.zeros_like(w)
        center[:, :, 4] = w.sum(-1)
        central = w - center
        angular = .5 * (w - w[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]])
        # Keep intensity-sensitive channels alongside derivative channels.
        idx = torch.arange(w.shape[0], device=w.device) % 4
        z = torch.where((idx == 1)[:, None, None], central, w)
        z = torch.where((idx == 2)[:, None, None], angular, z)
        return z.view_as(self.weight)

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'),
                        self.effective_weight(), self.bias, self.stride, groups=self.groups)


class Block(nn.Module):
    def __init__(self, ci, co, stride=1, difference=True):
        super().__init__()
        self.dw = DifferenceConv(ci, ci, stride, ci, difference)
        self.pw = nn.Conv2d(ci, co, 1)

    def forward(self, x):
        return F.relu(self.pw(F.relu(self.dw(x))))


class NativeContourNet(nn.Module):
    def __init__(self, difference=True):
        super().__init__()
        self.stem = DifferenceConv(3, 8, difference=difference)
        self.native = Block(8, 8, difference=difference)
        self.down1 = Block(8, 16, 2, difference)
        self.half_block = Block(16, 16, difference=difference)
        self.down2 = Block(16, 32, 2, difference)
        self.context = nn.ModuleList([Block(32, 32, difference=difference) for _ in range(3)])
        self.reduce = nn.Conv2d(32, 16, 1)
        self.decode1 = Block(16, 8, difference=difference)
        self.finish = DifferenceConv(8, 8, groups=8, difference=difference)
        self.edge = nn.Conv2d(8, 1, 1)
        nn.init.constant_(self.edge.bias, -1.)

    def forward(self, x):
        n = self.native(F.relu(self.stem(x)))
        h = self.half_block(self.down1(n))
        c = self.down2(h)
        for block in self.context:
            c = c + block(c)
        h = self.decode1(F.interpolate(F.relu(self.reduce(c)), scale_factor=2, mode='nearest') + h)
        n = F.interpolate(h, scale_factor=2, mode='nearest') + n
        return self.edge(F.relu(self.finish(n)))  # Unclipped signed logits.

    def folded(self):
        net = copy.deepcopy(self)
        with torch.no_grad():
            for layer in net.modules():
                if isinstance(layer, DifferenceConv):
                    layer.weight.copy_(layer.effective_weight())
                    layer.difference = False
        return net

    def budget(self):
        rows = []
        handles = []
        def record(name):
            def hook(m, inp, y):
                rows.append(dict(name=name, ci=m.in_channels, co=m.out_channels,
                    kernel=m.kernel_size[0], groups=m.groups, height=y.shape[-2], width=y.shape[-1],
                    weights=m.weight.numel(), bias=m.bias.numel(),
                    macs=y.shape[-2]*y.shape[-1]*m.weight.numel()))
            return hook
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d): handles.append(m.register_forward_hook(record(name)))
        with torch.no_grad(): self(torch.zeros(1, 3, 720, 1280, device=next(self.parameters()).device))
        for h in handles: h.remove()
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    convolution_weights=sum(r['weights'] for r in rows),
                    ideal_macs=sum(r['macs'] for r in rows), operators=rows,
                    board_validated=False, complete_style_network=False,
                    output='native 720p one-channel contour logits; sigmoid is display/training only')
