"""Audited FP32 unpaired photo-to-game CUT-style training experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .cut_wide_model import Discriminator, Generator, ResidualGenerator, PatchProjection, patch_nce
from .data import audit_style, load_content_manifest
from .engine import set_reproducible
from .losses import PerceptualLoss
from .runtime import HERE, configure_workspace, file_hash, write_json

MODEL = HERE / "cut_wide_model.py"


def image_tensor(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return ((torch.from_numpy(pixels).permute(2, 0, 1).float() - 128.0) / 128.0).unsqueeze(0)


def sample_images(content, style, style_manifest, rng, config):
    kinds = {row["file"]: row["kind"] for row in style_manifest["records"]}
    grouped = {kind: [item for item in style if kinds[Path(item["path"]).name] == kind]
               for kind in ("wide_scene", "texture_region")}
    policy = style_manifest["sampling_policy"]
    wide = rng.random() < policy["wide_scene_probability"]
    kind = "wide_scene" if wide else "texture_region"
    source = rng.choice(content)
    reference = rng.choice(grouped[kind])
    with Image.open(source["path"]) as file:
        photo = file.convert("RGB")
    with Image.open(reference["path"]) as file:
        game = file.convert("RGB")
    if wide:
        # Full 16:9 views carry scene-wide lighting and spatial layout.
        size = (config["wide_width"], config["wide_height"])
        photo = photo.resize(size, Image.Resampling.BICUBIC)
        game = game.resize(size, Image.Resampling.BICUBIC)
    else:
        px = rng.randrange(photo.width - 256 + 1)
        py = rng.randrange(photo.height - 256 + 1)
        gx = rng.randrange(game.width - 256 + 1)
        gy = rng.randrange(game.height - 256 + 1)
        photo = photo.crop((px, py, px + 256, py + 256))
        game = game.crop((gx, gy, gx + 256, gy + 256))
    if rng.random() < 0.5:
        photo = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        game = game.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image_tensor(photo).cuda(), image_tensor(game).cuda(), {
        "content": source["path"], "style": reference["path"], "kind": kind,
        "shape": list(photo.size[::-1]),
    }


@torch.no_grad()
def visual_review(generator, records, output, step):
    folder = output / f"validation_{step:06d}"
    folder.mkdir(exist_ok=False)
    generator.eval()
    # One early and one late scene from each normal-focus source.
    by_source = {}
    for row in records:
        by_source.setdefault(row["source"], []).append(row)
    chosen = []
    for rows in by_source.values():
        chosen.extend([rows[len(rows) // 4], rows[3 * len(rows) // 4]])
    entries = []
    for i, row in enumerate(chosen):
        with Image.open(row["path"]) as image:
            input_image = image_tensor(image).cuda()
        rendered = generator(input_image).clamp(-1, 127 / 128)
        before = ((input_image[0] * 128 + 128).round().clamp(0, 255).byte()
                  .permute(1, 2, 0).cpu().numpy())
        after = ((rendered[0] * 128 + 128).round().clamp(0, 255).byte()
                 .permute(1, 2, 0).cpu().numpy())
        from PIL import Image as PILImage
        path = folder / f"comparison_{i:02d}.png"
        PILImage.fromarray(np.concatenate([before, after], axis=1)).save(path)
        entries.append({"input": row["path"], "comparison": str(path),
                        "sha256": file_hash(path)})
    write_json(folder / "review.json", {"status": "VISUAL_REVIEW_REQUIRED",
                                        "step": step, "images": entries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "cut_wide_config.json")
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Run outside isolated workspace")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["schema"] != "cut-wide-fp32-v1" or not 0 < args.stop_after <= config["maximum_steps"]:
        raise ValueError("Invalid configuration")
    content = load_content_manifest(config["content_manifest"])
    style = audit_style(Path(config["style_manifest"]).parent)
    style_data = json.loads(Path(config["style_manifest"]).read_text(encoding="utf-8"))
    train = [r for r in content["images"] if r["split"] == "train"]
    validation = [r for r in content["images"] if r["split"] == "val"]
    expected_style_count = config.get("expected_style_count", 42)
    if len(train) < 100 or len(validation) < 40 or len(style) != expected_style_count:
        raise ValueError("Reviewed data coverage changed")
    set_reproducible(config["seed"])
    rng = random.Random(config["seed"])
    generator_type = {"resnet": Generator, "residual": ResidualGenerator}.get(
        config.get("generator_arch", "resnet"))
    if generator_type is None:
        raise ValueError("Unknown generator architecture")
    generator = generator_type(width=config["generator_width"],
                               blocks=config["residual_blocks"]).cuda()
    discriminator = Discriminator(width=config["discriminator_width"]).cuda()
    projector = PatchProjection((config["generator_width"],
                                 config["generator_width"] * 2,
                                 *([config["generator_width"] * 4] * 3)),
                                projection=config["projection_width"]).cuda()
    perceptual = None
    if config.get("content_loss", "patch_nce") == "vgg_relu3":
        perceptual = PerceptualLoss(root).cuda().eval()
    elif config.get("content_loss", "patch_nce") != "patch_nce":
        raise ValueError("Unknown content loss")
    optimizer_g = torch.optim.Adam(list(generator.parameters()) + list(projector.parameters()),
                                   lr=config["learning_rate"], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"],
                                   betas=(0.5, 0.999))
    binding = {"trainer_sha256": file_hash(Path(__file__)), "model_sha256": file_hash(MODEL),
               "config_sha256": file_hash(args.config),
               "content_sha256": file_hash(config["content_manifest"]),
               "style_sha256": file_hash(config["style_manifest"]),
               "torch_version": str(torch.__version__),
               "vgg_sha256": perceptual.weights_sha256 if perceptual else None}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["binding"] != binding or state["config"] != config:
            raise ValueError("Checkpoint source/data/config mismatch")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        projector.load_state_dict(state["projector"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        rng.setstate(state["sampler_rng"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
        if args.stop_after <= step:
            raise ValueError("Resume target must exceed checkpoint step")
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        (args.run / "source").mkdir()
        for path in (Path(__file__), MODEL, args.config):
            shutil.copy2(path, args.run / "source" / path.name)
        write_json(args.run / "run.json", {"binding": binding, "config": config,
                                           "train_count": len(train),
                                           "validation_count": len(validation),
                                           "style_count": len(style),
                                           "quality_gate": "UNREVIEWED"})
        step = 0

    def checkpoint():
        if (file_hash(Path(__file__)) != binding["trainer_sha256"]
                or file_hash(MODEL) != binding["model_sha256"]
                or file_hash(args.config) != binding["config_sha256"]):
            raise ValueError("Training source changed during run")
        path = args.run / f"checkpoint_{step:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"binding": binding, "config": config, "step": step,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "projector": projector.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "sampler_rng": rng.getstate(), "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
        os.replace(temporary, path)
        return path

    log = args.run / f"metrics_from_{step:06d}_{time.time_ns()}.jsonl"
    status = "RUNNING"
    last_checkpoint = args.resume
    try:
        while step < args.stop_after:
            generator.train()
            discriminator.train()
            started = time.perf_counter()
            real_a, real_b, spec = sample_images(train, style, style_data, rng, config)
            fake_b = generator(real_a)
            optimizer_d.zero_grad(set_to_none=True)
            pred_real = discriminator(real_b)
            pred_fake = discriminator(fake_b.detach())
            d_loss = 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real))
                            + F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
            d_loss.backward()
            optimizer_d.step()
            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            prediction = discriminator(fake_b)
            gan_loss = F.mse_loss(prediction, torch.ones_like(prediction))
            if perceptual is None:
                nce_loss = patch_nce(generator, projector, real_a, fake_b,
                                     patches=config["nce_patches"])
                feature_loss = torch.zeros((), device="cuda")
            else:
                with torch.no_grad():
                    source_feature = perceptual.extract(real_a)["relu3_3"]
                feature_loss = F.l1_loss(perceptual.extract(fake_b)["relu3_3"],
                                         source_feature)
                nce_loss = torch.zeros((), device="cuda")
            lowpass_loss = F.l1_loss(F.avg_pool2d(fake_b, 16),
                                     F.avg_pool2d(real_a, 16))
            edge_loss = (F.l1_loss(fake_b[:, :, 1:] - fake_b[:, :, :-1],
                                   real_a[:, :, 1:] - real_a[:, :, :-1])
                         + F.l1_loss(fake_b[:, :, :, 1:] - fake_b[:, :, :, :-1],
                                     real_a[:, :, :, 1:] - real_a[:, :, :, :-1]))
            (gan_loss + 0.5 * config["nce_weight"] * nce_loss
             + config.get("vgg_content_weight", 0.0) * feature_loss
             + config["lowpass_weight"] * lowpass_loss
             + config["edge_weight"] * edge_loss).backward()
            identity = generator(real_b)
            if perceptual is None:
                identity_nce = patch_nce(generator, projector, real_b, identity,
                                         patches=config["nce_patches"])
                identity_pixel = torch.zeros((), device="cuda")
                (0.5 * config["nce_weight"] * identity_nce).backward()
            else:
                identity_nce = torch.zeros((), device="cuda")
                identity_pixel = F.l1_loss(identity, real_b)
                (config.get("identity_weight", 0.0) * identity_pixel).backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                list(generator.parameters()) + list(projector.parameters()), 10.0,
                error_if_nonfinite=True)
            optimizer_g.step()
            discriminator.requires_grad_(True)
            step += 1
            row = {"step": step, "gan": gan_loss.item(), "nce": nce_loss.item(),
                   "identity_nce": identity_nce.item(), "discriminator": d_loss.item(),
                   "vgg_content": feature_loss.item(), "identity_pixel": identity_pixel.item(),
                   "lowpass": lowpass_loss.item(), "edge": edge_loss.item(),
                   "gradient": gradient.item(), "seconds": time.perf_counter() - started,
                   "sample": spec}
            if any(not np.isfinite(v) for v in (row["gan"], row["nce"], row["identity_nce"],
                                                      row["discriminator"], row["lowpass"],
                                                      row["edge"], row["gradient"],
                                                      row["vgg_content"], row["identity_pixel"])):
                raise FloatingPointError("Nonfinite training metric")
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if step == 1 or step % 10 == 0 or step == args.stop_after:
                write_json(args.run / "progress.json", {"status": "RUNNING", "step": step,
                                                       "losses": {k: row[k] for k in
                                                                  ("gan", "nce", "identity_nce", "discriminator", "lowpass", "edge", "vgg_content", "identity_pixel")},
                                                       "seconds_last_step": row["seconds"]})
            if step in config["validation_steps"]:
                visual_review(generator, validation, args.run, step)
            if step % config["checkpoint_every"] == 0 or step == args.stop_after:
                last_checkpoint = checkpoint()
        status = "STOP_AFTER_REACHED"
    finally:
        write_json(args.run / "status.json", {"status": status, "step": step,
                                               "quality_gate": "UNREVIEWED",
                                               "last_checkpoint": str(last_checkpoint) if last_checkpoint else None})


if __name__ == "__main__":
    main()
