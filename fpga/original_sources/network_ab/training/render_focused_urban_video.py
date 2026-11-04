"""Five-second diagnostic teacher video; native source beside 768px generated view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       UniPCMultistepScheduler)

from .probe_diffusion_teacher import canny_control, resize_wide
from .render_focused_urban_probe import NEGATIVE, PROMPT
from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-clip", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-sparse", type=Path)
    args = parser.parse_args()
    root = configure_workspace()
    clip = json.loads((args.source_clip.parent / "clip.json").read_text(encoding="utf-8"))
    if (clip["status"] != "selected_visual_review" or clip["frames"] != 150 or
            file_hash(args.source_clip) != clip["output_sha256"]):
        raise ValueError("Focused source clip changed")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"schema": "focused-urban-five-second-video-v1",
              "source_clip": str(args.source_clip), "source_sha256": file_hash(args.source_clip),
              "lora_weights": str(args.weights), "lora_sha256": file_hash(args.weights),
              "indices": list(range(0, 150, 3)), "fps": 10, "strength": 0.5,
              "steps": 32, "control_scale": 0.8, "guidance_scale": 7.0,
              "prompt": PROMPT, "negative": NEGATIVE, "seed": 20261002,
              "resolution_note": "Teacher view is 768x432; this diagnostic video does not claim native 1080p output."}
    write_json(args.output_dir / "config.json", config)
    sparse = None
    if args.reuse_sparse:
        sparse = json.loads((args.reuse_sparse / "config.json").read_text(encoding="utf-8"))
        if (sparse["source_sha256"] != config["source_sha256"] or
                sparse["lora_sha256"] != config["lora_sha256"] or
                sparse["prompt"] != PROMPT or sparse["negative"] != NEGATIVE or
                sparse["steps"] != config["steps"] or sparse["seed"] != config["seed"] or
                0.5 not in sparse["strengths"]):
            raise ValueError("Sparse probe does not match dense video configuration")
    preflight = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight / "controlnet_canny.json").read_text(encoding="utf-8"))
    network = ControlNetModel.from_pretrained(
        control["snapshot"], variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model["snapshot"], controlnet=network, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.weights.parent, weight_name=args.weights.name,
                           adapter_name="game")
    pipe.set_adapters("game", adapter_weights=1.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    capture = cv2.VideoCapture(str(args.source_clip))
    if not capture.isOpened():
        raise RuntimeError("Cannot open focused clip")
    video = args.output_dir / "source_vs_focused_teacher_5s.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (1536, 432))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot open video writer")
    completed = []
    try:
        for index in range(150):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Missing frame {index}")
            if index % 3:
                continue
            source = resize_wide(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), 768)
            style_path = args.output_dir / f"styled_{index:03d}.png"
            reuse = (args.reuse_sparse / f"styled_s50_{index:03d}.png"
                     if sparse and index in sparse["indices"] else None)
            if reuse is not None and reuse.is_file():
                styled = Image.open(reuse).convert("RGB")
            else:
                with torch.inference_mode():
                    styled = pipe(
                        prompt=PROMPT, negative_prompt=NEGATIVE,
                        image=source, control_image=canny_control(source),
                        strength=0.5, controlnet_conditioning_scale=0.8,
                        guidance_scale=7.0, num_inference_steps=32,
                        generator=torch.Generator(device="cpu").manual_seed(20261002),
                    ).images[0]
            if styled.size != (768, 432):
                raise ValueError(f"Wrong styled size at {index}")
            styled.save(style_path)
            panel = Image.new("RGB", (1536, 432))
            panel.paste(source, (0, 0))
            panel.paste(styled, (768, 0))
            writer.write(cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
            completed.append({"index": index, "styled_sha256": file_hash(style_path)})
            if len(completed) % 5 == 0:
                write_json(args.output_dir / "status.json", {
                    **config, "status": "rendering", "frames_done": len(completed)})
                print(f"video frames {len(completed)}/50", flush=True)
    finally:
        capture.release()
        writer.release()
    check = cv2.VideoCapture(str(video))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != 50:
        raise RuntimeError(f"Diagnostic video has {decoded}, not 50 frames")
    write_json(args.output_dir / "status.json", {
        **config, "status": "rendered_visual_review_required",
        "frames_done": 50, "video": str(video), "video_sha256": file_hash(video),
        "completed": completed})
    print(video)


if __name__ == "__main__":
    main()
