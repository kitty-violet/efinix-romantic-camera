"""Isolated AdaIN image stylizer, matching the public PyTorch-AdaIN tensors.

Architecture follows https://github.com/naoto0804/pytorch-AdaIN/blob/master/net.py
The local copy is small and explicit so third-party Python is never executed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def _conv(c_in, c_out, kernel=3):
    return nn.Conv2d(c_in, c_out, kernel)


def encoder():
    layers = [_conv(3, 3, 1)]
    widths = ((3, 64), (64, 64), (64, 128), (128, 128),
              (128, 256), (256, 256), (256, 256), (256, 256),
              (256, 512))
    for index, (c_in, c_out) in enumerate(widths):
        if index in (2, 4, 8):
            layers.append(nn.MaxPool2d(2, 2, 0, ceil_mode=True))
        layers.extend((nn.ReflectionPad2d(1), _conv(c_in, c_out), nn.ReLU()))
    return nn.Sequential(*layers)


def decoder():
    layers = []
    widths = ((512, 256), (256, 256), (256, 256), (256, 256),
              (256, 128), (128, 128), (128, 64), (64, 64), (64, 3))
    for index, (c_in, c_out) in enumerate(widths):
        if index in (1, 5, 7):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.extend((nn.ReflectionPad2d(1), _conv(c_in, c_out)))
        if index != len(widths) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained(root):
    root = Path(root)
    records = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    if {Path(row["file"]).name for row in records} != {"vgg_normalised.pth", "decoder.pth"}:
        raise ValueError("Unexpected AdaIN release manifest")
    for row in records:
        if Path(row["file"]).resolve().parent != root.resolve() or _sha256(row["file"]) != row["sha256"]:
            raise ValueError("AdaIN asset path/hash mismatch")
    enc, dec = encoder(), decoder()
    # weights_only prevents execution of arbitrary Python objects from .pth.
    loaded = enc.load_state_dict(torch.load(root / "vgg_normalised.pth", map_location="cpu", weights_only=True),
                                 strict=False)
    if loaded.missing_keys:
        raise ValueError(f"AdaIN encoder tensor missing: {loaded.missing_keys}")
    dec.load_state_dict(torch.load(root / "decoder.pth", map_location="cpu", weights_only=True),
                        strict=True)
    return enc.eval().requires_grad_(False), dec.eval()


def channel_moments(value):
    flat = value.flatten(2)
    return flat.mean(-1).unsqueeze(-1).unsqueeze(-1), flat.var(-1, unbiased=True).add(1e-5).sqrt().unsqueeze(-1).unsqueeze(-1)


def style_transfer(enc, dec, content, style, alpha=1.0):
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in [0,1]")
    features = enc(content)
    target = enc(style)
    c_mean, c_std = channel_moments(features)
    s_mean, s_std = channel_moments(target)
    merged = (features-c_mean)/c_std*s_std+s_mean
    return dec(alpha*merged+(1-alpha)*features).clamp(0, 1)
