"""Render sparse LoRA keyframes for an actual five-second 1080p source clip."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       UniPCMultistepScheduler)

from .media import decode_frames, probe_video
from .probe_diffusion_teacher import canny_control, resize_wide
from .render_simple_stage_video import NEGATIVE, PROMPT
from .runtime import configure_workspace, file_hash, write_json


SPARSE_INDICES = [0, 30, 60, 90, 120, 147]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense", action="store_true",
                        help="Use every third source frame for the full five seconds")
    parser.add_argument("--reuse-dir", type=Path,
                        help="Reuse verified same-config source and style PNGs by hardlink")
    args = parser.parse_args()
    indices = list(range(0, 150, 3)) if args.dense else SPARSE_INDICES
    root = configure_workspace()
    state = json.loads(args.source_state.read_text(encoding="utf-8"))
    video = Path(state["source_video"])
    weights = Path(state["lora_weights"])
    if file_hash(video) != state["source_sha256"] or file_hash(weights) != state["lora_sha256"]:
        raise ValueError("Source video or LoRA changed")
    metadata = probe_video(video)
    if (metadata["width"], metadata["height"]) != (1920, 1080):
        raise ValueError("Expected a native 1080p video")
    preflight = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight / "controlnet_canny.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "five-second-keyframe-style-v1", "source_video": str(video),
        "source_sha256": state["source_sha256"], "lora_weights": str(weights),
        "lora_sha256": state["lora_sha256"], "indices": indices,
        "prompt": PROMPT, "negative_prompt": NEGATIVE,
        "strength": 0.7, "steps": 32, "control_scale": 0.8,
        "guidance_scale": 7.0, "seed": 20261001,
    }
    write_json(args.output_dir / "config.json", config)
    prepared = []
    for index, pts, pixels in decode_frames(video, indices=indices,
                                            hlg_exposure_stops=-1.0):
        original = ImageOps.fit(Image.fromarray(pixels), (1280, 720),
                                method=Image.Resampling.BICUBIC)
        source = resize_wide(original, 768)
        source_path = args.output_dir / f"source_{index:06d}.png"
        if source_path.is_file():
            if not np.array_equal(np.asarray(source),
                                  np.asarray(Image.open(source_path).convert("RGB"))):
                raise ValueError(f"Source frame {index} changed")
        else:
            source.save(source_path)
        if args.reuse_dir is not None:
            reuse_source = args.reuse_dir / f"source_{index:06d}.png"
            reuse_style = args.reuse_dir / f"styled_{index:06d}.png"
            target_style = args.output_dir / f"styled_{index:06d}.png"
            if reuse_source.is_file() and reuse_style.is_file() and not target_style.exists():
                if not np.array_equal(np.asarray(source),
                                      np.asarray(Image.open(reuse_source).convert("RGB"))):
                    raise ValueError(f"Reuse source mismatch at frame {index}")
                os.link(reuse_style, target_style)
        prepared.append((index, pts, source))
    if len(prepared) != len(indices):
        raise RuntimeError("Incomplete source decode")

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
    completed = []
    for index, pts, source in prepared:
        output = args.output_dir / f"styled_{index:06d}.png"
        if output.is_file():
            styled = Image.open(output).convert("RGB")
            if styled.size != source.size:
                raise ValueError(f"Wrong styled frame size {index}")
        else:
            with torch.inference_mode():
                styled = pipe(
                    prompt=PROMPT, negative_prompt=NEGATIVE,
                    image=source, control_image=canny_control(source),
                    strength=config["strength"],
                    controlnet_conditioning_scale=config["control_scale"],
                    guidance_scale=config["guidance_scale"],
                    num_inference_steps=config["steps"],
                    generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
                ).images[0]
            styled.save(output)
        completed.append({"index": index, "pts_seconds": pts,
                          "source_sha256": file_hash(args.output_dir / f"source_{index:06d}.png"),
                          "styled_sha256": file_hash(output)})
        write_json(args.output_dir / "status.json",
                   {**config, "status": "rendering", "completed": completed})
        print(f"keyframe {index} at {pts:.3f}s", flush=True)
    write_json(args.output_dir / "status.json",
               {**config, "status": "rendered_visual_review_required",
                "completed": completed})
    print(f"keyframes {len(completed)} complete", flush=True)


if __name__ == "__main__":
    main()
