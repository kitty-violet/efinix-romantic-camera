"""Train a project-specific SD1.5 LoRA from approved game frames.

This teacher is separate from the FPGA network. It learns only the target
appearance; real-content scene structure is supplied at inference by Canny.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, ControlNetModel, DDPMScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from .runtime import configure_workspace, file_hash, write_json


CAPTION = ("a dxgame screenshot of a photorealistic first-person survival "
           "video game, detailed weathered buildings, overgrown vegetation, "
           "realistic rough surface textures, warm natural daylight")
SEED = 20261001


def target_paths(manifest: Path, stage_selection: Path | None = None) -> list[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["status"] != "approved" or data["image_count"] != len(data["records"]):
        raise ValueError("Target manifest is not approved or internally consistent")
    rows = [row for row in data["records"]
            if row.get("lighting") == "daylight"
            or row.get("scene", "").startswith("daylight")]
    if len(rows) != 56:
        raise ValueError(f"Expected 56 approved daylight images, found {len(rows)}")
    if stage_selection is not None:
        stage = json.loads(stage_selection.read_text(encoding="utf-8"))
        accepted = {"simple-stage-target-selection-v1": "reviewed_for_simple_stage",
                    "focused-urban-target-selection-v1": "reviewed_focused_urban"}
        if (stage.get("schema") not in accepted or
                stage.get("status") != accepted[stage["schema"]] or
                stage.get("parent_manifest_sha256") != file_hash(manifest) or
                stage.get("image_count") != len(stage["records"])):
            raise ValueError("Invalid vehicle-light stage selection")
        by_name = {row["file"]: row for row in rows}
        if len(stage["records"]) != len({row["file"] for row in stage["records"]}):
            raise ValueError("Duplicate vehicle-light target")
        for selected in stage["records"]:
            original = by_name.get(selected["file"])
            if original is None or original["sha256"] != selected["sha256"]:
                raise ValueError(f"Stage target not in approved daylight set: {selected['file']}")
        rows = [by_name[selected["file"]] for selected in stage["records"]]
    paths = [manifest.parent / row["file"] for row in rows]
    for path, row in zip(paths, rows):
        if not path.is_file() or file_hash(path) != row["sha256"]:
            raise ValueError(f"Missing or modified target image: {path}")
        if Image.open(path).size != (1280, 720):
            raise ValueError(f"Unexpected target resolution: {path}")
    return paths


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def target_views(image: Image.Image, index: int,
                 safe_roi: tuple[int, int, int, int] | None = None) -> list[Image.Image]:
    # A full panorama teaches overall palette; native crops retain sharp textures.
    if safe_roi is not None:
        left, top, right, bottom = safe_roi
        if not (0 <= left < right <= image.width and
                0 <= top < bottom <= image.height and
                right - left >= 512 and bottom - top >= 512):
            raise ValueError(f"Invalid safe target ROI: {safe_roi}")
        image = image.crop(safe_roi)
    wide = image.resize((640, 360), Image.Resampling.LANCZOS)
    views = [wide]
    for view_no in range(2):
        rng = random.Random(SEED + index * 23 + view_no)
        x = rng.randrange(0, image.width - 512 + 1)
        y = rng.randrange(0, image.height - 512 + 1)
        views.append(image.crop((x, y, x + 512, y + 512)))
    return views


def cache_latents(model_path: Path, paths: list[Path],
                  rois: dict[str, tuple[int, int, int, int]] | None = None,
                  with_canny: bool = False) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True).to("cuda")
    vae.eval()
    latents = []
    controls = []
    with torch.no_grad():
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            for view in target_views(image, index,
                                     rois.get(path.name) if rois else None):
                pixels = image_tensor(view).to("cuda", dtype=torch.float16)
                latent = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor
                latents.append(latent[0].cpu().contiguous())
                if with_canny:
                    rgb = np.asarray(view.convert("RGB"))
                    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)
                    control = torch.from_numpy(edges.copy()).unsqueeze(0).repeat(3, 1, 1)
                    controls.append(control.to(dtype=torch.float16).div_(255).contiguous())
            if index % 10 == 0:
                print(f"latent cache {index + 1}/{len(paths)}", flush=True)
    del vae
    torch.cuda.empty_cache()
    return latents, controls


def save_state(run: Path, step: int, unet, optimizer, scaler, audit: dict) -> None:
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    save_dir = run / f"weights_{step:06d}"
    save_dir.mkdir(parents=True, exist_ok=False)
    StableDiffusionPipeline.save_lora_weights(
        save_dir, unet_lora_layers=lora, safe_serialization=True)
    torch.save({"step": step, "lora": get_peft_model_state_dict(unet),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "audit": audit}, run / f"checkpoint_{step:06d}.pt")
    write_json(run / "status.json", {
        **audit, "status": "training", "step": step,
        "weights": str(save_dir / "pytorch_lora_weights.safetensors"),
        "checkpoint": str(run / f"checkpoint_{step:06d}.pt")})
    print(f"checkpoint {step}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="game-style-lora-20261002-01")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-selection", type=Path,
                        help="Reviewed subset of approved target frames for a simpler first stage")
    parser.add_argument("--condition-on-canny", action="store_true",
                        help="Train LoRA with the same frozen Canny ControlNet used at inference")
    parser.add_argument("--caption", default=CAPTION,
                        help="Frozen appearance caption; recorded in the checkpoint audit")
    args = parser.parse_args()
    if not 1 <= args.steps <= 20000:
        raise ValueError("Invalid step count")
    root = configure_workspace()
    run = root / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    teacher = json.loads((root / "runs/diffusion-teacher-preflight/sd15.json").read_text(encoding="utf-8"))
    model_path = Path(teacher["snapshot"])
    manifest = root / "data/processed/style-target-20261001-v14-expanded-reviewed/dataset-manifest.json"
    paths = target_paths(manifest, args.target_selection)
    rois = {}
    if args.target_selection:
        stage = json.loads(args.target_selection.read_text(encoding="utf-8"))
        if stage["schema"] == "focused-urban-target-selection-v1":
            rois = {row["file"]: tuple(row["safe_roi_xyxy"])
                    for row in stage["records"]}
            if set(rois) != {path.name for path in paths}:
                raise ValueError("Incomplete safe ROI mapping")
    audit = {"schema": "game-style-lora-v1", "model_revision": teacher["revision"],
             "target_manifest": str(manifest), "target_manifest_sha256": file_hash(manifest),
             "target_count": len(paths), "view_count": len(paths) * 3,
             "caption": args.caption, "seed": SEED, "precision": "fp16-base-fp32-lora",
             "train_script_sha256": file_hash(Path(__file__))}
    if args.target_selection:
        audit["target_selection"] = str(args.target_selection.resolve())
        audit["target_selection_sha256"] = file_hash(args.target_selection)
    if args.condition_on_canny:
        control_info = json.loads((root / "runs/diffusion-teacher-preflight/controlnet_canny.json").read_text(encoding="utf-8"))
        audit["conditioning"] = "frozen-canny-controlnet-from-target-edges"
        audit["controlnet_revision"] = control_info["revision"]
        audit["control_scale"] = 0.8
    write_json(run / "config.json", audit)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(SEED)
    latents, controls = cache_latents(model_path, paths, rois, args.condition_on_canny)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                            variant="fp16", torch_dtype=torch.float16,
                                            use_safetensors=True, local_files_only=True).to("cuda")
    encoder.eval()
    with torch.no_grad():
        tokens = tokenizer(args.caption, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to("cuda")
        context = encoder(tokens)[0].detach()
    del encoder
    torch.cuda.empty_cache()

    controlnet = None
    if args.condition_on_canny:
        controlnet = ControlNetModel.from_pretrained(
            control_info["snapshot"], variant="fp16", torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True).to("cuda")
        controlnet.requires_grad_(False)
        controlnet.eval()

    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True).to("cuda")
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=8, lora_alpha=8, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    cast_training_params(unet, dtype=torch.float32)
    unet.enable_gradient_checkpointing()
    unet.train()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable or any(p.dtype != torch.float32 for p in trainable):
        raise RuntimeError("LoRA parameters must be FP32 and trainable")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["audit"] != audit:
            raise ValueError("Resume configuration/data/code drift")
        set_peft_model_state_dict(unet, state["lora"], adapter_name="default")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
    write_json(run / "status.json", {**audit, "status": "training", "step": start})
    loss_path = run / "losses.json"
    losses = (json.loads(loss_path.read_text(encoding="utf-8"))["losses"]
              if start and loss_path.is_file() else [])
    for step in range(start + 1, args.steps + 1):
        selector = random.Random(SEED + step)
        choice = selector.randrange(len(latents))
        latent = latents[choice].unsqueeze(0).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(SEED + step)
        noise = torch.randn(latent.shape, device="cuda", dtype=latent.dtype, generator=generator)
        timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                 (1,), device="cuda", generator=generator).long()
        noisy = noise_scheduler.add_noise(latent, noise, timestep)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if controlnet is not None:
                conditioning = controls[choice].unsqueeze(0).to("cuda")
                with torch.no_grad():
                    down, mid = controlnet(
                        noisy, timestep, context, controlnet_cond=conditioning,
                        conditioning_scale=audit["control_scale"], return_dict=False)
                prediction = unet(
                    noisy, timestep, context,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid).sample
            else:
                prediction = unet(noisy, timestep, context).sample
            loss = F.mse_loss(prediction.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        losses.append(value)
        if step <= 10 or step % 25 == 0:
            print(f"step {step}/{args.steps} loss={value:.5f} "
                  f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_state(run, step, unet, optimizer, scaler, audit)
            write_json(run / "losses.json", {"steps_completed": step, "losses": losses,
                                             "last_50_mean": float(np.mean(losses[-50:]))})
    write_json(run / "status.json", {**audit, "status": "trained_visual_review_required",
                                    "step": args.steps,
                                    "weights": str(run / f"weights_{args.steps:06d}/pytorch_lora_weights.safetensors"),
                                    "checkpoint": str(run / f"checkpoint_{args.steps:06d}.pt")})


if __name__ == "__main__":
    main()
