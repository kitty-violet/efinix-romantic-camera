"""WCT2 research teacher, using pinned official weights and native RGB input.

Network topology: unmodified NAVER WCT2 model.py (MIT). Whitening/coloring
math follows its utils/core.py (NVIDIA CC BY-NC-SA 4.0 attribution retained
in the downloaded source). This wrapper caches style covariance, not images.
It is an offline teacher, not the hardware network or a newly trained model.
"""
from pathlib import Path
import importlib.util
import json

import torch

from .runtime import file_hash

REVISION = "05d17c02c6eb78a3e0afdbc1cf1d7549d4ad1543"


def matrix_power_spd(matrix, power):
    # Covariances are symmetric PSD, with the author's unit regularizer.
    values, vectors = torch.linalg.eigh(matrix)
    if not torch.isfinite(values).all() or values.min() <= 0:
        raise FloatingPointError("Covariance is not finite positive definite")
    return (vectors * values.pow(power)[None]) @ vectors.T


def moments(feature):
    flat = feature[0].flatten(1)
    mean = flat.mean(1, keepdim=True)
    centered = flat - mean
    covariance = centered @ centered.T / max(flat.shape[1] - 1, 1)
    covariance += torch.eye(flat.shape[0], device=flat.device)
    return mean, covariance


def transform(feature, style, alpha):
    if alpha == 0:
        return feature
    mean, covariance = moments(feature)
    flat = feature[0].flatten(1)
    matrix = style["color"] @ matrix_power_spd(covariance, -.5)
    result = matrix @ (flat - mean) + style["mean"]
    result = result.clamp(flat.min(), flat.max()).reshape_as(feature)
    return result * alpha + feature * (1 - alpha)


class WaveletTeacher:
    def __init__(self, root, device="cuda", option="cat5"):
        self.device = device
        if option not in ("cat5", "sum"):
            raise ValueError(option)
        self.option = option
        self.directory = Path(root) / "cache/wct2" / REVISION
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        for row in self.manifest["files"]:
            if file_hash(row["path"]) != row["sha256"]:
                raise ValueError("Changed WCT2 asset: " + row["path"])
        spec = importlib.util.spec_from_file_location("wct2_pinned_model", self.directory / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.encoder = module.WaveEncoder(option).to(device).eval()
        self.decoder = module.WaveDecoder(option).to(device).eval()
        for name, net in (("encoder", self.encoder), ("decoder", self.decoder)):
            state = torch.load(self.directory / f"model_checkpoints/wave_{name}_{option}_l4.pth",
                               map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            net.requires_grad_(False)
        self.styles = {}

    @torch.no_grad()
    def prepare_style(self, name, image):
        stats, skips = {}, {}
        def save(key, value):
            mean, cov = moments(value)
            stats[key] = {"mean": mean, "color": matrix_power_spd(cov, .5)}
        value = image
        for level in (1, 2, 3, 4):
            value = self.encoder.encode(value, skips, level)
            save(f"encoder{level}", value)
        save("decoder4", value)
        for level in (4, 3, 2):
            value = self.decoder.decode(value, skips, level)
            save(f"decoder{level-1}", value)
        for level in (1, 2, 3):
            for component, feature in enumerate(skips[f"pool{level}"]):
                save(f"skip{level}_{component}", feature)
        self.styles[name] = stats
        return stats

    @torch.no_grad()
    def __call__(self, image, style=None, stages=(), alpha=.7, levels=(1, 2, 3, 4)):
        if image.shape[0] != 1 or image.shape[1] != 3 or any(d % 8 for d in image.shape[-2:]):
            raise ValueError("Expected batch-one RGB with dimensions divisible by eight")
        stats = self.styles.get(style)
        stages = set(stages)
        if stages and stats is None:
            raise ValueError("Style not prepared")
        skips, value = {}, image
        for level in (1, 2, 3, 4):
            value = self.encoder.encode(value, skips, level)
            if "encoder" in stages and level in levels:
                value = transform(value, stats[f"encoder{level}"], alpha)
        if "skip" in stages:
            for level in (1, 2, 3):
                skips[f"pool{level}"] = [transform(v, stats[f"skip{level}_{i}"], alpha)
                                          for i, v in enumerate(skips[f"pool{level}"])]
        for level in (4, 3, 2, 1):
            if "decoder" in stages and level in levels and (level != 4 or "encoder" not in stages):
                value = transform(value, stats[f"decoder{level}"], alpha)
            value = self.decoder.decode(value, skips, level)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite WCT2 output")
        return value.clamp(0, 1)
