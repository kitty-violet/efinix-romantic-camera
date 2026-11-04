"""Small CUT-style generator, PatchGAN discriminator, and patch contrastive loss.

Architecture adapted for the existing CUDA runtime from the official CUT idea:
https://github.com/taesungp/contrastive-unpaired-translation
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    def __init__(self, width=32, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, width, 7),
                                  nn.InstanceNorm2d(width), nn.ReLU(True))
        self.down1 = nn.Sequential(nn.Conv2d(width, width * 2, 3, 2, 1),
                                   nn.InstanceNorm2d(width * 2), nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 3, 2, 1),
                                   nn.InstanceNorm2d(width * 4), nn.ReLU(True))
        self.middle = nn.ModuleList(Residual(width * 4) for _ in range(blocks))
        self.up1 = nn.Sequential(nn.Conv2d(width * 4, width * 2, 3, padding=1),
                                 nn.InstanceNorm2d(width * 2), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1),
                                 nn.InstanceNorm2d(width), nn.ReLU(True))
        self.output = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(width, 3, 7), nn.Tanh())

    def encode(self, x):
        a = self.stem(x)
        b = self.down1(a)
        c = self.down2(b)
        x = c
        intermediate = None
        for i, block in enumerate(self.middle):
            x = block(x)
            if i == 2:
                intermediate = x
        return [a, b, c, intermediate, x]

    def decode(self, bottleneck):
        x = F.interpolate(bottleneck, scale_factor=2, mode="nearest")
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.output(self.up2(x))

    def forward(self, x):
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("Image height and width must be divisible by four")
        return self.decode(self.encode(x)[-1])


class ResidualGenerator(Generator):
    """Bounded image residual with an exact identity initialization.

    The adversarial branch can change local appearance, while the input skip
    preserves scene geometry and prevents wholesale sky/material replacement.
    """
    def __init__(self, width=32, blocks=6, residual_scale=0.5):
        super().__init__(width=width, blocks=blocks)
        if not 0 < residual_scale <= 1:
            raise ValueError("Residual scale must be in (0, 1]")
        self.residual_scale = residual_scale
        nn.init.zeros_(self.output[1].weight)
        nn.init.zeros_(self.output[1].bias)

    def forward(self, x):
        delta = super().forward(x)
        return (x + self.residual_scale * delta).clamp(-1, 127 / 128)


class Discriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        layers = [nn.Conv2d(3, width, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        channels = width
        for stride, out in ((2, width * 2), (2, width * 4), (1, width * 8)):
            layers += [nn.Conv2d(channels, out, 4, stride, 1),
                       nn.InstanceNorm2d(out), nn.LeakyReLU(0.2, True)]
            channels = out
        layers.append(nn.Conv2d(channels, 1, 4, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PatchProjection(nn.Module):
    def __init__(self, channels=(32, 64, 128, 128, 128), projection=128):
        super().__init__()
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(c, projection), nn.ReLU(True),
                                                 nn.Linear(projection, projection))
                                   for c in channels)

    def forward(self, features, indices):
        result = []
        for layer, feature, index in zip(self.heads, features, indices):
            flat = feature.flatten(2).transpose(1, 2)
            picked = flat.index_select(1, index)
            result.append(F.normalize(layer(picked), dim=-1))
        return result


def patch_nce(generator, projector, original, generated, patches=64, temperature=0.07):
    """Positive is same image location; all other sampled locations are negatives."""
    key_features = generator.encode(original)
    query_features = generator.encode(generated)
    indices = [torch.randperm(q.shape[-2] * q.shape[-1], device=q.device)[:patches]
               for q in query_features]
    keys = projector([k.detach() for k in key_features], indices)
    queries = projector(query_features, indices)
    losses = []
    for query, key in zip(queries, keys):
        positive = (query * key).sum(-1, keepdim=True)
        negatives = torch.bmm(query, key.transpose(1, 2))
        size = negatives.shape[-1]
        negatives = negatives.masked_fill(torch.eye(size, device=negatives.device,
                                                    dtype=torch.bool)[None], -1e4)
        logits = torch.cat((positive, negatives), dim=-1) / temperature
        losses.append(F.cross_entropy(logits.flatten(0, 1),
                                      torch.zeros(logits.shape[0] * size,
                                                  dtype=torch.long, device=logits.device)))
    return torch.stack(losses).mean()
