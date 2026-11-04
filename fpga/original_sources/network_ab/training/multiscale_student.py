"""Software candidate: local detail skips plus a large-context encoder/decoder.

Not a B1-compatible coefficient image. Bilinear feature resizing and the new
graph require independent deployment work. No global/spatial normalization.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1, activate=True):
        super().__init__(nn.ReplicationPad2d(kernel//2) if kernel > 1 else nn.Identity(),
            nn.Conv2d(cin, cout, kernel, stride, groups=groups, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU() if activate else nn.Identity())


class Separable(nn.Sequential):
    def __init__(self, cin, cout, stride=1, activate=True):
        super().__init__(ConvBN(cin, cin, 3, stride, cin), ConvBN(cin, cout, 1, activate=activate))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(Separable(channels, channels), Separable(channels, channels, activate=False))

    def forward(self, x):
        return F.relu(x + self.layers(x)*.5)


class MultiscaleStudent(nn.Module):
    WIDTHS = {'small': (8, 16, 24, 32), 'medium': (12, 24, 48, 64)}

    def __init__(self, size='small'):
        super().__init__()
        self.size = size
        a, b, c, d = self.WIDTHS[size]
        self.stem = ConvBN(3, a)
        self.down = nn.ModuleList([Separable(a, b, 2), Separable(b, c, 2), Separable(c, d, 2)])
        self.context = nn.Sequential(*[Residual(d) for _ in range(4)])
        self.project = nn.ModuleList([ConvBN(d, c, 1), ConvBN(c, b, 1), ConvBN(b, a, 1)])
        self.fuse = nn.ModuleList([Separable(c, c), Separable(b, b), Separable(a, a)])
        self.head = nn.Conv2d(a, 3, 3, padding=0)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start near identity while every branch has nonzero output gradients.
        nn.init.normal_(self.head.weight, std=.002)
        self.eval()  # Running BN statistics are frozen; affine remains trainable.

    def forward(self, x):
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError('Dimensions must be divisible by eight')
        feats = [self.stem(x)]
        for layer in self.down:
            feats.append(layer(feats[-1]))
        y = self.context(feats[-1])
        for project, fuse, skip in zip(self.project, self.fuse, reversed(feats[:-1])):
            y = F.interpolate(project(y), size=skip.shape[-2:], mode='bilinear', align_corners=False)
            y = fuse(y + skip)
        correction = self.head(F.pad(y, (1, 1, 1, 1), mode='replicate'))
        return (x + correction).clamp(-1, 1)


def profile(net, x):
    rows = []
    def capture(name):
        def hook(module, args, output):
            rows.append(dict(name=name, shape=list(output.shape), kernel=list(module.kernel_size),
                             stride=list(module.stride), groups=module.groups,
                             weights=module.weight.numel(), macs=output.numel()*module.weight[0].numel()))
        return hook
    hooks = [m.register_forward_hook(capture(n)) for n, m in net.named_modules() if isinstance(m, nn.Conv2d)]
    with torch.inference_mode():
        net(x)
    for h in hooks:
        h.remove()
    return dict(size=net.size, parameters=sum(p.numel() for p in net.parameters()),
                convolution_count=len(rows), convolution_macs=sum(r['macs'] for r in rows), convolutions=rows,
                scope='Software mathematical profile, excludes resize/add/clamp; not hardware timing or fit.',
                normalization='Frozen BN, no per-frame/global normalization',
                rgb_bypass='Learned residual prediction inside the network; no output blending knob')
