"""Training-only style teacher. Never export this graph as an FPGA model.

The deployed student remains the unmodified B contract. This deliberately
larger encoder/decoder tests whether a less constrained generator can produce
an acceptable target before any attempt to distil its behavior.
"""
from __future__ import annotations

import torch
from torch import nn


def block(in_channels, out_channels, *, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.ReLU(inplace=False),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
        )

    def forward(self, value):
        return value + self.layers(value)


class TrainingOnlyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, 9, bias=False),
            nn.InstanceNorm2d(32, affine=True, track_running_stats=False),
            nn.ReLU(inplace=False),
        )
        self.down1 = block(32, 64, stride=2)
        self.down2 = block(64, 128, stride=2)
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        self.up1 = block(128, 64)
        self.up2 = block(64, 32)
        self.head = nn.Sequential(nn.ReflectionPad2d(4), nn.Conv2d(32, 3, 9))
        # Begin close to identity while leaving gradients for the full teacher.
        nn.init.normal_(self.head[1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2] % 4 or image.shape[-1] % 4:
            raise ValueError("NCHW RGB with H/W divisible by four required")
        full = self.stem(image)
        half = self.down1(full)
        quarter = self.residuals(self.down2(half))
        value = torch.nn.functional.interpolate(quarter, scale_factor=2, mode="nearest")
        value = self.up1(value) + half
        value = torch.nn.functional.interpolate(value, scale_factor=2, mode="nearest")
        value = self.up2(value) + full
        return (image + self.head(value).tanh()).clamp(-1.0, 127.0 / 128.0)
