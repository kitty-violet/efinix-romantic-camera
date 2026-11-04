"""Bounded reconstruction/BN probes. Diagnostic weights are never deployment weights."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from .runtime import HERE, configure_workspace, fingerprints, snapshot_sources, write_json


def initialize_rgb_path(model):
    """Initialize existing operators as a signed RGB path; do not add/replace ops.

    Six ReLU lanes carry +RGB/-RGB through stride-2 sampling and nearest repeat.
    Zero final residual BN gamma makes each existing residual initially identity.
    Running statistics are identity; use frozen running statistics for this probe.
    """
    import torch
    from torch import nn
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.zero_()
                module.running_var.fill_(1-module.eps)
                module.num_batches_tracked.zero_()
                module.weight.fill_(1)
                module.bias.zero_()
        for name, sequence in model.operators.items():
            conv = next(m for m in sequence if isinstance(m, nn.Conv2d))
            if name.startswith("res"):
                if name.endswith("conv2__pw"):
                    next(m for m in sequence if isinstance(m, nn.BatchNorm2d)).weight.zero_()
                continue
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
            if name == "stem":
                for out in range(conv.out_channels):
                    lane = out % 6
                    conv.weight[out, lane % 3, 1, 1] = 1 if lane < 3 else -1
            elif name == "rgb_head":
                for channel in range(3):
                    conv.weight[channel, channel, 0, 0] = 1
                    conv.weight[channel, channel+3, 0, 0] = -1
            elif conv.groups == conv.in_channels:
                conv.weight[:, 0, 1, 1] = 1
            else:
                for out in range(conv.out_channels):
                    conv.weight[out, out % 6, 0, 0] = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--mode", choices=["matrix", "bn"], default="matrix")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cross-controls", action="store_true")
    args = parser.parse_args()
    root = configure_workspace()
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from network_ab.contract_torch import build_fp32_model
    from .data import image_tensor, load_content_manifest
    from .engine import set_reproducible
    from .evaluate import to_u8
    from .losses import PerceptualLoss

    if (root / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Diagnostic output outside approved runs")
    args.run.mkdir(parents=True, exist_ok=False)
    binding = fingerprints()
    snapshot_sources(args.run / "source", binding)
    config = json.loads((HERE / "fp32_config.json").read_text(encoding="utf-8"))
    set_reproducible(config["seed"])
    manifest = load_content_manifest(args.manifest)
    train = [r for r in manifest["images"] if r["split"] == "train"]
    val = [r for r in manifest["images"] if r["split"] == "val"]
    # Selection is deterministic and recorded; test sources are never opened here.
    chosen = min((r for r in train if "080118" in r["source"]),
                 key=lambda r: abs(r["timestamp"]-14))
    criterion = PerceptualLoss(root).cuda()
    write_json(args.run / "plan.json", {"purpose": "diagnostic_only", "binding": binding,
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "fixed_training_image": chosen, "config": config})
    if args.mode == "bn":
        if not args.checkpoint:
            raise ValueError("BN probe requires existing checkpoint")
        model = build_fp32_model().cuda()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
        frozen = copy.deepcopy(model.state_dict())
        records = [chosen, train[0], val[9], val[22]]
        versions = {}
        for mode in ["saved_eval", "image_batch_stats", "recalibrated_eval"]:
            model.load_state_dict(frozen)
            if mode == "recalibrated_eval":
                model.train()
                with torch.no_grad():
                    for record in train:
                        model(image_tensor(record["path"], "cuda").unsqueeze(0))
            model.train(mode == "image_batch_stats")
            rows, images = [], []
            with torch.no_grad():
                for record in records:
                    x = image_tensor(record["path"], "cuda").unsqueeze(0)
                    y = model(x)
                    lc = F.mse_loss(criterion.extract(y)["relu2_2"], criterion.extract(x)["relu2_2"])
                    rows.append({"path": record["path"], "content_loss": lc.item(),
                                 "pixel_mse": F.mse_loss(x,y).item(), "std": y.std().item()})
                    images.append(to_u8(y[0]))
            versions[mode] = {"rows": rows, "images": images}
        for i, record in enumerate(records):
            original = to_u8(image_tensor(record["path"]))
            Image.fromarray(np.concatenate([original]+[v["images"][i] for v in versions.values()],axis=1)).save(args.run/f"bn_{i}.png")
        write_json(args.run / "result.json", {"purpose": "diagnostic_only", "columns": ["input"]+list(versions),
                   "versions": {k:v["rows"] for k,v in versions.items()}})
        return

    xfull = image_tensor(chosen["path"], "cuda").unsqueeze(0)
    x = F.interpolate(xfull, size=(180,320), mode="area")
    with torch.no_grad():
        target = criterion.extract(x)["relu2_2"]
    results = []
    cases = ([("default","frozen","content"), ("rgb_path","batch","content")]
             if args.cross_controls else [("default","batch","pixel"),
                                          ("default","batch","content"),
                                          ("rgb_path","frozen","content")])
    for init, normalization, objective in cases:
        torch.manual_seed(config["seed"])
        model = build_fp32_model().cuda()
        if init == "rgb_path":
            initialize_rgb_path(model)
            model.eval()
            with torch.no_grad():
                expected = x[:,:,::4,::4].repeat_interleave(4,-2).repeat_interleave(4,-1)
                identity_error = (model(x)-expected).abs().max().item()
            if identity_error > 1e-5:
                raise RuntimeError(f"RGB path initialization mismatch: {identity_error}")
        model.train(normalization == "batch")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        name = f"{init}_{normalization}_{objective}"
        directory = args.run / name
        directory.mkdir()
        history = []
        started = time.perf_counter()
        for step in range(args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            y = model(x)
            lc = F.mse_loss(criterion.extract(y)["relu2_2"],target)
            pixel = F.mse_loss(y,x)
            loss = pixel if objective == "pixel" else lc
            if not torch.isfinite(loss):
                raise FloatingPointError(name)
            if step % 50 == 0 or step == args.steps:
                row = {"step": step, "content_loss": lc.item(), "pixel_mse": pixel.item()}
                history.append(row)
                print(name, row, flush=True)
                Image.fromarray(np.concatenate([to_u8(x[0]),to_u8(y[0])],axis=1)).save(directory/f"step_{step:04d}.png")
                write_json(args.run / "progress.json", {"case": name, **row})
            if step < args.steps:
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            yf = model(xfull)
        Image.fromarray(np.concatenate([to_u8(xfull[0]),to_u8(yf[0])],axis=1)).save(directory/"full720_eval.png")
        torch.save({"purpose":"diagnostic_only", "model":model.state_dict()}, directory/"diagnostic.pt")
        results.append({"case":name,"history":history,"seconds":time.perf_counter()-started})
    write_json(args.run / "result.json", {"purpose":"diagnostic_only", "results":results})


if __name__ == "__main__":
    main()
