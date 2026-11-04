"""Render fixed sparse frames from the focused SDR content clip with one LoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       UniPCMultistepScheduler)

from .probe_diffusion_teacher import canny_control, resize_wide
from .runtime import configure_workspace, file_hash, write_json


PROMPT = ("a dxgame screenshot of a photorealistic first-person survival video "
          "game in an abandoned underground station, weathered concrete walls, "
          "stained plaster, aged stone stairs, corroded metal railings, "
          "rough masonry texture, strong directional light and deep shadows")
NEGATIVE = ("cartoon, anime, painting, low resolution, blurry, white speckles, "
            "glitter, glowing particles, text, HUD, crosshair, interface, "
            "extra limbs, deformed person")
INDICES = (0, 30, 60, 90, 120, 149)
STRENGTHS = (0.5, 0.7)


def read_sdr_frame(video: Path, index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise RuntimeError("Cannot seek source")
        ok, bgr = capture.read()
        if not ok or bgr is None or bgr.shape[:2] != (1080, 1920):
            raise RuntimeError(f"Missing native SDR frame {index}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-clip", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = configure_workspace()
    meta = json.loads((args.source_clip.parent / "clip.json").read_text(encoding="utf-8"))
    if (meta["status"] != "selected_visual_review" or
            meta["output_sha256"] != file_hash(args.source_clip) or
            meta["frames"] != 150 or meta["duration_seconds"] != 5.0):
        raise ValueError("Focused source clip is not verified")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"schema": "focused-urban-sparse-probe-v1",
              "source_clip": str(args.source_clip), "source_sha256": file_hash(args.source_clip),
              "lora_weights": str(args.weights), "lora_sha256": file_hash(args.weights),
              "indices": list(INDICES), "strengths": list(STRENGTHS),
              "prompt": PROMPT, "negative": NEGATIVE, "steps": 32,
              "control_scale": 0.8, "guidance_scale": 7.0, "seed": 20261002}
    write_json(args.output_dir / "config.json", config)
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
    results = []
    for index in INDICES:
        source = resize_wide(read_sdr_frame(args.source_clip, index), 768)
        source.save(args.output_dir / f"source_{index:03d}.png")
        control_image = canny_control(source)
        board = Image.new("RGB", (768 * 3, 456), "#151515")
        board.paste(source, (0, 24))
        draw = ImageDraw.Draw(board)
        draw.text((12, 5), f"SOURCE {index / 30:.1f}s", fill="white")
        for n, strength in enumerate(STRENGTHS, 1):
            with torch.inference_mode():
                styled = pipe(
                    prompt=PROMPT, negative_prompt=NEGATIVE,
                    image=source, control_image=control_image,
                    strength=strength, controlnet_conditioning_scale=0.8,
                    guidance_scale=7.0, num_inference_steps=32,
                    generator=torch.Generator(device="cpu").manual_seed(20261002),
                ).images[0]
            path = args.output_dir / f"styled_s{int(strength * 100):02d}_{index:03d}.png"
            styled.save(path)
            board.paste(styled, (768 * n, 24))
            draw.text((768 * n + 12, 5), f"LoRA strength {strength:.2f}", fill="white")
            results.append({"index": index, "strength": strength,
                            "output": str(path), "sha256": file_hash(path)})
        board.save(args.output_dir / f"review_{index:03d}.jpg", quality=95)
        write_json(args.output_dir / "status.json", {
            **config, "status": "rendering", "results": results})
        print(f"review frame {index} complete", flush=True)
    write_json(args.output_dir / "status.json", {
        **config, "status": "rendered_visual_review_required", "results": results})


if __name__ == "__main__":
    main()
