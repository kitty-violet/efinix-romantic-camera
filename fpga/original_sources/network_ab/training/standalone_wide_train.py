"""Train an independent FP32 game-style model with wide-scene-first references."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from .runtime import HERE, configure_workspace, file_hash, write_json


def weighted_style_targets(criterion, style, style_manifest):
    """Average VGG Gram targets by scene type, then by native image."""
    import torch
    from .data import image_tensor
    from .losses import gram

    records = {r["file"]: r for r in style_manifest["records"]}
    policy = style_manifest["sampling_policy"]
    counts = {kind: sum(r["kind"] == kind for r in records.values())
              for kind in ("wide_scene", "texture_region")}
    if min(counts.values()) == 0 or abs(sum(policy.values()) - 1.0) > 1e-8:
        raise ValueError("Invalid wide/texture target weighting")
    accumulated = None
    with torch.no_grad():
        for item in style:
            kind = records[Path(item["path"]).name]["kind"]
            p = policy[kind + "_probability"] / counts[kind]
            tensor = image_tensor(item["path"], "cuda").unsqueeze(0)
            current = {key: gram(value) * p for key, value in criterion.extract(tensor).items()}
            accumulated = current if accumulated is None else {
                key: accumulated[key] + current[key] for key in current}
            del tensor, current
    return accumulated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "teacher_probe_config.json")
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Teacher run must remain inside workspace/runs")

    import torch
    from .data import Sampler, audit_style, image_tensor, load_content_manifest
    from .engine import set_reproducible
    from .evaluate import render_video, validate_images
    from .losses import PerceptualLoss
    from .teacher_probe_model import TrainingOnlyTeacher

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config["schema"] != "standalone-wide-game-style-v1"
            or config["purpose"] != "software-game-style-reproduction"
            or config["crop_hw"] != [[256, 256]]
            or config.get("style_target_mode", "sampled_image") not in ("sampled_image", "mean_gram")
            or not 0 < args.stop_after <= config["maximum_steps"]):
        raise ValueError("Unreviewed teacher probe configuration")
    manifest_path = Path(config["content_manifest"])
    style_manifest = Path(config["style_manifest"])
    content = load_content_manifest(manifest_path)
    style = audit_style(style_manifest.parent)
    style_data = json.loads(style_manifest.read_text(encoding="utf-8"))
    if (len(style) != style_data["image_count"] or style_data["wide_scene_count"] < 10
            or len([r for r in content["images"] if r["split"] == "val"]) < 20):
        raise ValueError("Expected reviewed wide style and normal-focus validation")
    if any(r["sha256"] in {s["sha256"] for s in style} for r in content["images"]):
        raise ValueError("Style/content image leakage")
    set_reproducible(config["seed"])
    model = TrainingOnlyTeacher().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    sampler = Sampler([r for r in content["images"] if r["split"] == "train"], style, config)
    criterion = PerceptualLoss(root).cuda()
    binding = {
        "purpose": "standalone_software_game_style_weights",
        "config_sha256": file_hash(args.config),
        "content_manifest_sha256": file_hash(manifest_path),
        "style_manifest_sha256": file_hash(style_manifest),
        "teacher_model_sha256": file_hash(HERE / "teacher_probe_model.py"),
        "teacher_runner_sha256": file_hash(Path(__file__)),
        "vgg_sha256": criterion.weights_sha256,
        "torch_version": str(torch.__version__),
    }
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["binding"] != binding or state["config"] != config or state["purpose"] != binding["purpose"]:
            raise ValueError("Teacher checkpoint/source/data/config mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
        if args.stop_after <= step:
            raise ValueError("Stop-after must exceed resumed step")
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        source = args.run / "source"
        source.mkdir()
        for path in (Path(__file__), HERE / "teacher_probe_model.py", args.config):
            shutil.copy2(path, source / path.name)
        step = 0
        write_json(args.run / "run.json", {
            "binding": binding, "config": config,
            "teacher_parameters": sum(p.numel() for p in model.parameters()),
            "style_count": len(style), "content_train_count": len(sampler.records),
            "validation_count": len([r for r in content["images"] if r["split"] == "val"]),
            "style_sampling_policy": style_data["sampling_policy"],
            "quality_gate": "UNREVIEWED",
            "deployable": "software_fp32_only",
        })

    def save_checkpoint():
        # These guard the running source; old runs remain reproducible from their snapshots.
        if (file_hash(HERE / "teacher_probe_model.py") != binding["teacher_model_sha256"]
                or file_hash(Path(__file__)) != binding["teacher_runner_sha256"]
                or file_hash(args.config) != binding["config_sha256"]):
            raise ValueError("Teacher source/config changed during training")
        path = args.run / f"checkpoint_{step:06d}.pt"
        temporary = args.run / f"checkpoint_{step:06d}.tmp"
        state = {
            "purpose": binding["purpose"], "step": step, "binding": binding,
            "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        torch.save(state, temporary)
        os.replace(temporary, path)
        return path

    log = args.run / f"metrics_from_{step:06d}_{time.time_ns()}.jsonl"
    target_grams = (weighted_style_targets(criterion, style, style_data)
                    if config.get("style_target_mode") == "mean_gram" else None)
    status = "RUNNING"
    try:
        while step < args.stop_after:
            model.train()
            started = time.perf_counter()
            spec = sampler.next_spec(step)
            source_image = image_tensor(spec["content"]["path"], "cuda")
            y, x, h, w = spec["crop"]
            image = source_image[:, y:y+h, x:x+w]
            if spec["flip"]:
                image = image.flip(-1)
            image = image.unsqueeze(0)
            if target_grams is None:
                target = image_tensor(spec["style"]["path"], "cuda").unsqueeze(0)
                content_target, style_target = criterion.targets(image, target)
            else:
                with torch.no_grad():
                    content_target = criterion.extract(image)["relu2_2"]
                style_target = target_grams
            optimizer.zero_grad(set_to_none=True)
            result = model(image)
            content_loss, style_loss = criterion(result, (content_target, style_target))
            loss = content_loss + config["style_weight"] * style_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite teacher loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite teacher weights")
            step += 1
            record = {
                "step": step, "content_loss": content_loss.detach().item(),
                "style_loss": style_loss.detach().item(),
                "weighted_style_loss": config["style_weight"] * style_loss.detach().item(),
                "gradient_before_clip": gradient.item(), "seconds": time.perf_counter() - started,
                "sample": spec,
            }
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if step == 1 or step % 25 == 0 or step == args.stop_after:
                write_json(args.run / "progress.json", {
                    "status": "RUNNING", "completed_steps": step,
                    "content_loss": record["content_loss"],
                    "weighted_style_loss": record["weighted_style_loss"],
                    "seconds_last_step": record["seconds"],
                })
            if step in config["validation_steps"]:
                if target_grams is None:
                    target_grams = weighted_style_targets(criterion, style, style_data)
                validation = [r for r in content["images"] if r["split"] == "val"]
                selected = validation[::max(1, len(validation)//12)][:12] if step < 500 else validation
                directory = args.run / f"validation_{step:06d}"
                report = validate_images(model, criterion, target_grams, selected,
                                         directory, config["style_weight"])
                ratios = [r["style_loss"] / r["input_style_loss"] for r in report["images"]]
                write_json(directory / "review.json", {
                    "validation_count": len(ratios), "mean_style_ratio": sum(ratios) / len(ratios),
                    "improved_count": sum(r < 1 for r in ratios),
                    "mean_content_loss": report["mean_content_loss"],
                    "status": "METRICS_ONLY_REQUIRES_VISUAL_REVIEW",
                })
                videos = [v for v in content["videos"] if v["split"] == "val"]
                if videos:
                    write_json(directory / "video.json", render_video(
                        model, videos[0]["path"], directory / "comparison.mp4",
                        seconds=4 if step < 500 else 6))
            if step % 250 == 0 or step == args.stop_after:
                save_checkpoint()
        status = "STOP_AFTER_REACHED"
    finally:
        write_json(args.run / "status.json", {
            "status": status, "completed_steps": step,
            "quality_gate": "UNREVIEWED", "deployable": "software_fp32_only",
            "last_complete_checkpoint": str(args.run / f"checkpoint_{step:06d}.pt")
            if (args.run / f"checkpoint_{step:06d}.pt").exists() else None,
        })


if __name__ == "__main__":
    main()
