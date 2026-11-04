"""Q038-Q040 frozen VGG19 losses; no download and no additional loss terms."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import file_hash
from .optimization_policy import set_optimization_mode

FEATURES = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 24}


def gram(value):
    n, c, h, w = value.shape
    flat = value.reshape(n, c, h*w)
    return torch.bmm(flat, flat.transpose(1, 2)) / (c*h*w)


def rgb01(value):
    return (128.0 * value + 128.0) / 255.0


def color_moments(value):
    """Per-image RGB mean and standard deviation, without a new network op."""
    rgb = rgb01(value)
    mean = rgb.mean((2, 3))
    std = (rgb.var((2, 3), unbiased=False) + 1e-6).sqrt()
    return torch.cat((mean, std), dim=1)


def color_moment_loss(output, target_moments):
    return F.mse_loss(color_moments(output), target_moments)


class PerceptualLoss(nn.Module):
    def __init__(self, root):
        super().__init__()
        # Build official feature topology only, avoiding the unused 123M-parameter classifier.
        from torchvision.models.vgg import cfgs, make_layers
        root = Path(root)
        manifest = json.loads((root / "runs/environment/vgg19-weights.json").read_text(encoding="utf-8"))
        path = root / "cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth"
        if file_hash(path) != manifest["sha256"] or not manifest["sha256"].startswith("dcbb9e9d"):
            raise ValueError("VGG19 weights hash changed")
        features = make_layers(cfgs["E"], batch_norm=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        features.load_state_dict({k.removeprefix("features."): v for k, v in state.items()
                                  if k.startswith("features.")}, strict=True)
        self.features = features[:max(FEATURES.values())+1].eval().requires_grad_(False)
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.weights_sha256 = manifest["sha256"]
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def extract(self, image):
        value = (rgb01(image) - self.mean) / self.std
        result = {}
        for index, layer in enumerate(self.features):
            value = layer(value)
            for name, position in FEATURES.items():
                if index == position:
                    result[name] = value
        return result

    def targets(self, content, style):
        with torch.no_grad():
            target_content = self.extract(content)["relu2_2"]
            target_style = {k: gram(v) for k, v in self.extract(style).items()}
        return target_content, target_style

    def forward(self, output, targets):
        expected_content, expected_style = targets
        features = self.extract(output)
        content = F.mse_loss(features["relu2_2"], expected_content)
        style = torch.stack([F.mse_loss(gram(features[k]), expected_style[k]) for k in FEATURES]).mean()
        return content, style


def calibrate_weight(criterion, output, targets, ratio=2.0):
    probe = output.detach().clone().requires_grad_(True)
    content, style = criterion(probe, targets)
    gc, = torch.autograd.grad(content, probe, retain_graph=True)
    gs, = torch.autograd.grad(style, probe)
    nc, ns = gc.norm().item(), gs.norm().item()
    if not torch.isfinite(torch.tensor([nc, ns])).all() or min(nc, ns) <= 0:
        raise RuntimeError("Nonfinite/zero calibration gradient; no arbitrary weight fallback")
    return {"style_weight": ratio*nc/ns, "content_gradient_l2": nc,
            "style_gradient_l2": ns, "target_ratio": ratio,
            "achieved_ratio": (ratio*nc/ns)*ns/nc}


def calibrate_first_batch(model, criterion, sampler, config):
    """Measure the actual training batch, then restore every consumed state.

    Eval-mode output with untrained BN is nearly constant and is not the signal
    seen by the optimizer. Sum squared output-gradient norms across the same
    four microbatches (including their different crop sizes), then account for
    effective-batch averaging. No model/BN/RNG/sampler update survives this probe.
    """
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sampler_state = sampler.state_dict()
    previous_modes = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    parts, specs = [], []
    count = config["effective_batch"]
    try:
        set_optimization_mode(model, config)
        for _ in range(count):
            content, style, spec = sampler.next(0, "cuda")
            with torch.no_grad():
                output = model(content)
            parts.append(calibrate_weight(criterion, output, criterion.targets(content, style),
                                          config["style_content_gradient_ratio"]))
            specs.append(spec)
    finally:
        model.load_state_dict(state, strict=True)
        for module, mode in previous_modes:
            module.training = mode
        sampler.load_state_dict(sampler_state)
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    nc = math.sqrt(sum(p["content_gradient_l2"]**2 for p in parts))/count
    ns = math.sqrt(sum(p["style_gradient_l2"]**2 for p in parts))/count
    ratio = config["style_content_gradient_ratio"]
    weight = ratio*nc/ns
    return {"method": "training_mode_effective_batch_output_gradient_l2",
            "bn_policy": config.get("bn_policy", "batch_running_stats"),
            "style_weight": weight, "content_gradient_l2": nc, "style_gradient_l2": ns,
            "target_ratio": ratio, "achieved_ratio": weight*ns/nc,
            "effective_batch": count, "microbatches": parts, "first_batch": specs,
            "model_bn_sampler_rng_restored": True}
