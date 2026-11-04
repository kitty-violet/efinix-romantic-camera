"""Render a short source/teacher video from the no-vehicle park holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       UniPCMultistepScheduler)

from .media import decode_frames
from .probe_diffusion_teacher import canny_control, resize_wide
from .render_game_teacher_video import write_video
from .runtime import configure_workspace, file_hash, write_json


PROMPT = ("a dxgame screenshot of a tree-lined park walkway with people, "
          "first-person survival video game rendering, detailed foliage "
          "and path textures, natural daylight")
NEGATIVE = ("cartoon, anime, oil painting, watercolor, flat colors, "
            "blurry, oversmoothed, low resolution, text, HUD, interface")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        help="LoRA safetensors; defaults to the original 500-step run")
    parser.add_argument("--output-dir", type=Path,
                        help="Separate output directory for this weight's video")
    args = parser.parse_args()
    root = configure_workspace()
    run = root / "runs/game-style-lora-simple-20261002-01"
    out = args.output_dir or root / "runs/simple-scene-stage-20261002/temporal-validation"
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "runs/simple-scene-stage-20261002/target-selection.json").read_text(encoding="utf-8"))
    content = json.loads((root / "data/processed/content-20261001-v4-minus1ev-draft/manifest.approved.json").read_text(encoding="utf-8"))
    anchor = next(row for row in content["images"] if
                  row["path"] == selection["fixed_unseen_content"])
    if anchor["split"] != "val" or file_hash(Path(anchor["path"])) != anchor["sha256"]:
        raise ValueError("Vehicle-free holdout image changed or leaked")
    source_video = Path(anchor["source"])
    if file_hash(source_video) != anchor["source_sha256"]:
        raise ValueError("Holdout video changed")
    preflight = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight / "controlnet_canny.json").read_text(encoding="utf-8"))
    weights = args.weights or run / "weights_000500/pytorch_lora_weights.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    status_path = out / "status.json"
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("lora_sha256") != file_hash(weights):
            raise ValueError("Output directory contains frames from different weights")
    params = {
        "schema": "simple-stage-consecutive-video-v1",
        "source_video": str(source_video), "source_sha256": file_hash(source_video),
        "source_frame_indices": list(range(0, 48, 3)), "fps": 10,
        "anchor": anchor["path"], "anchor_sha256": anchor["sha256"],
        "lora_weights": str(weights), "lora_sha256": file_hash(weights),
        "model_revision": model["revision"], "control_revision": control["revision"],
        "prompt": PROMPT, "strength": 0.7, "steps": 32,
        "control_scale": 0.8, "guidance": 7.0, "seed": 20261001,
        "warning": "Exploratory teacher clip; visual acceptance is required.",
    }
    write_json(out / "config.json", params)
    prepared = []
    for index, pts, pixels in decode_frames(
            source_video, indices=params["source_frame_indices"], hlg_exposure_stops=-1.0):
        original = ImageOps.fit(Image.fromarray(pixels), (1280, 720),
                                method=Image.Resampling.BICUBIC)
        if index == 0 and not np.array_equal(
                np.asarray(original), np.asarray(Image.open(anchor["path"]).convert("RGB"))):
            raise RuntimeError("Decoded first frame differs from approved holdout")
        source = resize_wide(original, 768)
        source.save(out / f"source_{index:06d}.png")
        prepared.append((index, pts, source))
    if len(prepared) != len(params["source_frame_indices"]):
        raise RuntimeError("Missing holdout video frames")

    network = ControlNetModel.from_pretrained(
        control["snapshot"], variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model["snapshot"], controlnet=network, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(weights.parent, weight_name=weights.name, adapter_name="game")
    pipe.set_adapters("game", adapter_weights=1.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    panels = []
    for i, (index, pts, source) in enumerate(prepared):
        result_path = out / f"styled_{index:06d}.png"
        if result_path.is_file():
            styled = Image.open(result_path).convert("RGB")
        else:
            with torch.inference_mode():
                styled = pipe(
                    prompt=PROMPT, negative_prompt=NEGATIVE,
                    image=source, control_image=canny_control(source),
                    strength=params["strength"],
                    controlnet_conditioning_scale=params["control_scale"],
                    guidance_scale=params["guidance"],
                    num_inference_steps=params["steps"],
                    generator=torch.Generator(device="cpu").manual_seed(params["seed"]),
                ).images[0]
            styled.save(result_path)
        panel = Image.new("RGB", (1536, 456), "#151515")
        panel.paste(source, (0, 24))
        panel.paste(styled, (768, 24))
        ImageDraw.Draw(panel).text(
            (12, 5), f"held-out source {index} ({pts:.3f}s)  |  500-step game teacher", fill="white")
        if i == 0:
            panel.save(out / "review_frame_000000.jpg", quality=95)
        panels.append(panel)
        write_json(out / "status.json", {**params, "status": "rendering",
                                          "frames_done": i + 1,
                                          "frames_total": len(prepared)})
        print(f"frame {i + 1}/{len(prepared)} {index}", flush=True)
    video = out / "source_vs_teacher.mp4"
    write_video(video, panels, params["fps"])
    write_json(out / "status.json", {**params,
                                      "status": "rendered_visual_review_required",
                                      "frames_done": len(prepared), "video": str(video),
                                      "video_sha256": file_hash(video)})
    print(video, flush=True)


if __name__ == "__main__":
    main()
