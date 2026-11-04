"""Audited real-content FP32 runner. No synthetic-data fallback or implicit downloads."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .runtime import HERE, assert_fingerprints, canonical_hash, configure_workspace, file_hash, fingerprints, snapshot_sources, write_json


def write_progress(path, value):
    """A locked UI telemetry file must not discard otherwise valid training."""
    try:
        write_json(path, value)
    except PermissionError as exc:
        if getattr(exc, "winerror", None) not in (5, 32, 33):
            raise
        print(f"PROGRESS_UPDATE_SKIPPED_AFTER_RETRIES: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpu-report", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "fp32_config.json")
    # Limit an invocation without altering the frozen 100k-step schedule.
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Run output must be under the authorized workspace/runs")
    import torch
    from .data import Sampler, audit_style, load_content_manifest
    from .engine import make_training, restore_checkpoint, save_checkpoint, set_reproducible, train_step
    from .evaluate import mean_color_targets, mean_style_targets, render_video, validate_images
    from .losses import PerceptualLoss, calibrate_first_batch
    config = json.loads(args.config.read_text(encoding="utf-8"))
    color_weight = config.get("color_moment_weight", 0.0)
    expected_extras = ["rgb_mean_std_color_moment_v1"] if color_weight else []
    if (config["micro_batch"] != 1 or config["effective_batch"] != 4
            or config["extra_losses"] != expected_extras):
        raise ValueError("Current trainer supports effective batch4 and the declared color-moment experiment only")
    if not isinstance(color_weight, (int, float)) or not math.isfinite(color_weight) or not 0 <= color_weight <= 1000:
        raise ValueError("Color-moment experiment weight must be a finite nonnegative scalar <=1000")
    gate = json.loads(args.gpu_report.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_DIAGNOSTIC_ONLY":
        raise ValueError("Actual GPU diagnostic gate required")
    assert_fingerprints(gate["fingerprints"])
    if gate["config_sha256"] != file_hash(args.config):
        raise ValueError("Configuration differs from the measured GPU gate")
    if not all(gate["training_sizes"][f"{h}x{w}"]["passed"] for h, w in config["crop_hw"]):
        raise ValueError("Configured crop size has not passed GPU backward")
    if config["full_frame_every"] and not gate["training_sizes"]["720x1280"]["passed"]:
        raise ValueError("720p backward did not pass; explicitly revise/validate schedule first")
    manifest = load_content_manifest(args.manifest)
    style = audit_style()
    style_digests = {r["sha256"] for r in style}
    if any(r["sha256"] in style_digests for r in manifest["images"]):
        raise ValueError("Style reference copied into content dataset")
    binding = {"fingerprints": fingerprints(), "data_manifest_sha256": file_hash(args.manifest),
               "style_manifest_sha256": canonical_hash(style), "gpu_report_sha256": file_hash(args.gpu_report)}
    if args.resume:
        if not args.run.is_dir() or not (args.run / "run.json").exists():
            raise ValueError("Resume requires an existing run")
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        snapshot_sources(args.run / "source", binding["fingerprints"])
    set_reproducible(config["seed"])
    model, optimizer, scheduler = make_training(config)
    sampler = Sampler([r for r in manifest["images"] if r["split"] == "train"], style, config)
    criterion = PerceptualLoss(root).cuda()
    binding["vgg_sha256"] = criterion.weights_sha256
    step = 0
    control = {"best_validation_objective": None, "unimproved_validations": 0}
    if args.resume:
        state = restore_checkpoint(args.resume, model, optimizer, scheduler, sampler,
                      binding=binding, config=config, purpose="fp32_training")
        step, calibration, control = state["completed_steps"], state["calibration"], state["control"]
    else:
        calibration = calibrate_first_batch(model, criterion, sampler, config)
        write_json(args.run / "run.json", {"binding": binding, "config": config, "calibration": calibration,
                   "manifest": manifest, "status": "RUNNING", "precision": "FP32; TF32 and AMP disabled"})
    target_grams = mean_style_targets(criterion, style)
    target_colors = mean_color_targets(style) if color_weight else None
    limit = min(config["max_steps"], args.stop_after or config["max_steps"])
    if limit <= step:
        raise ValueError("Stop-after must exceed restored step")
    last_complete = None
    status = "RUNNING"
    write_json(args.run / "status.json", {"status": status, "completed_steps_in_memory": step,
               "last_complete_checkpoint": str(args.resume or ""), "quality_gate": "UNREVIEWED",
               "quantization_allowed": False})
    log = args.run / f"metrics_from_{step:06d}_{time.time_ns()}.jsonl"
    try:
        while step < limit:
            assert_fingerprints(binding["fingerprints"])
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            metrics = train_step(model, optimizer, scheduler, criterion, sampler, step,
                                 calibration["style_weight"], config)
            torch.cuda.synchronize()
            step += 1
            metrics.update(seconds=time.perf_counter()-started,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False)+"\n")
            if step == 1 or step % 25 == 0 or step == limit:
                write_progress(args.run / "progress.json", {"status": "RUNNING", "completed_steps": step,
                    "invocation_limit": limit, "maximum_schedule_steps": config["max_steps"],
                    "content_loss": metrics["content_loss"], "weighted_style_loss": metrics["weighted_style_loss"],
                    "seconds_last_step": metrics["seconds"], "updated_unix_seconds": time.time()})
            if step % config["validate_every"] == 0 or step == 1:
                directory = args.run / f"validation_{step:06d}"
                validation = validate_images(model, criterion, target_grams,
                       [r for r in manifest["images"] if r["split"] == "val"], directory, calibration["style_weight"],
                       color_target=target_colors, color_weight=color_weight)
                videos = [v for v in manifest["videos"] if v["split"] == "val"]
                if videos:
                    write_json(directory / "video.json", render_video(model, videos[0]["path"], directory / "comparison.mp4", seconds=12))
                objective = validation["validation_objective"]
                improved = control["best_validation_objective"] is None or objective < control["best_validation_objective"]
                control["unimproved_validations"] = 0 if improved else control["unimproved_validations"]+1
                if improved:
                    control["best_validation_objective"] = objective
                    save_checkpoint(args.run / "best.pt", model, optimizer, scheduler, sampler,
                       step=step, binding=binding, config=config, calibration=calibration,
                       purpose="fp32_training", control=control)
            # An additional small recovery snapshot limits work lost to host I/O
            # failures; it does not change optimizer/sampler/loss/schedule state.
            if step % config["checkpoint_every"] == 0 or step % 250 == 0 or step == 1 or step == limit:
                last_complete = args.run / f"checkpoint_{step:06d}.pt"
                save_checkpoint(last_complete, model, optimizer, scheduler, sampler, step=step,
                   binding=binding, config=config, calibration=calibration, purpose="fp32_training", control=control)
            if control["unimproved_validations"] >= config["early_stopping_validations"]:
                status = "EARLY_STOPPED_REQUIRES_QUALITY_REVIEW"
                break
        if status == "RUNNING":
            status = "STOP_AFTER_REACHED" if step < config["max_steps"] else "MAX_STEPS_REACHED"
        last_complete = args.run / f"checkpoint_{step:06d}.pt"
        save_checkpoint(last_complete, model, optimizer, scheduler, sampler, step=step,
             binding=binding, config=config, calibration=calibration, purpose="fp32_training", control=control)
    except BaseException as exc:
        status = f"INTERRUPTED: {type(exc).__name__}: {exc}"
        # Do not save half-consumed sampler/RNG as an exactly resumable checkpoint.
        raise
    finally:
        final_status = {"status": status, "completed_steps_in_memory": step,
                   "last_complete_checkpoint": str(last_complete) if last_complete else str(args.resume or ""),
                   "quality_gate": "UNREVIEWED", "quantization_allowed": False}
        write_json(args.run / "status.json", final_status)
        write_progress(args.run / "progress.json", {**final_status, "completed_steps": step,
                   "updated_unix_seconds": time.time()})


if __name__ == "__main__":
    main()
