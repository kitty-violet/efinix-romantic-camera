"""FP32 steps and complete atomic checkpoints, shared by diagnostic and real runs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile

import torch

from network_ab.contract_torch import build_fp32_model
from .runtime import assert_fingerprints, atomic_replace, contract_guard, file_hash
from .optimization_policy import initialize_model, set_optimization_mode
from .losses import color_moment_loss, color_moments


def set_reproducible(seed):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ReplicationPad CUDA backward may use atomic additions. Report numerical
    # resume error; do not falsely promise bit-identical GPU training.
    torch.use_deterministic_algorithms(True, warn_only=True)


def learning_rate_factor(index, config):
    warmup, total = config["warmup_steps"], config["max_steps"]
    if index < warmup:
        return (index + 1) / warmup
    floor = config["minimum_learning_rate"] / config["learning_rate"]
    progress = min(1.0, max(0.0, (index-warmup) / max(1, total-warmup-1)))
    return floor + (1-floor)*0.5*(1+math.cos(math.pi*progress))


def make_training(config, device="cuda"):
    contract, _ = contract_guard(config["contract_sha256"])
    model = build_fp32_model(contract).to(device=device, dtype=torch.float32)
    initialize_model(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: learning_rate_factor(index, config))
    return model, optimizer, scheduler


def train_step(model, optimizer, scheduler, criterion, sampler, step, style_weight, config):
    set_optimization_mode(model, config)
    optimizer.zero_grad(set_to_none=True)
    content_sum = style_sum = color_sum = 0.0
    sample_specs = []
    for _ in range(config["effective_batch"]):
        content, style, spec = sampler.next(step, "cuda")
        targets = criterion.targets(content, style)
        output = model(content)
        lc, ls = criterion(output, targets)
        color_weight = config.get("color_moment_weight", 0.0)
        lcolor = color_moment_loss(output, color_moments(style).detach()) if color_weight else output.new_zeros(())
        loss = (lc + style_weight*ls + color_weight*lcolor) / config["effective_batch"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        content_sum += lc.detach().item() / config["effective_batch"]
        style_sum += ls.detach().item() / config["effective_batch"]
        color_sum += lcolor.detach().item() / config["effective_batch"]
        sample_specs.append(spec)
    squared = 0.0
    for name, value in model.named_parameters():
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
        squared += value.grad.detach().double().square().sum().item()
    if squared <= 0:
        raise FloatingPointError("All model gradients are zero")
    used_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Nonfinite model state after optimizer step")
    return {"step": step+1, "content_loss": content_sum, "style_loss": style_sum,
            "weighted_style_loss": style_weight*style_sum,
            "color_moment_loss": color_sum, "weighted_color_moment_loss": config.get("color_moment_weight", 0.0)*color_sum,
            "gradient_l2": math.sqrt(squared), "learning_rate": used_lr,
            "samples": sample_specs}


def save_checkpoint(path, model, optimizer, scheduler, sampler, *, step, binding,
                    config, calibration, purpose, control=None):
    assert_fingerprints(binding["fingerprints"])
    if purpose not in ("diagnostic_only", "fp32_training"):
        raise ValueError("Explicit checkpoint purpose required")
    state = {"schema": "model-training-checkpoint-v1", "purpose": purpose,
             "completed_steps": step, "binding": binding, "config": config,
             "calibration": calibration, "control": control or {},
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(),
             "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(temporary, path)
    return file_hash(path)


def restore_checkpoint(path, model, optimizer, scheduler, sampler, *, binding, config, purpose):
    assert_fingerprints(binding["fingerprints"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("schema") != "model-training-checkpoint-v1" or state["binding"] != binding
            or state["config"] != config or state["purpose"] != purpose):
        raise ValueError("Checkpoint data/code/config/purpose mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
