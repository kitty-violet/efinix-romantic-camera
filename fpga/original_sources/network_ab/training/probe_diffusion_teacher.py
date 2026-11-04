"""Evaluate a pinned SD1.5/Canny teacher on held-out, clear real-world scenes.

The base model has not learned this project's game footage. Outputs are probes,
not accepted style-transfer artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       UniPCMultistepScheduler)

from .runtime import configure_workspace, file_hash, write_json


PROMPT = ("photorealistic first-person 3D survival video game environment, "
          "detailed weathered materials, realistic surface textures, "
          "volumetric daylight, high quality game rendering")
NEGATIVE = ("cartoon, anime, oil painting, watercolor, flat colors, "
            "blurry, oversmoothed, low resolution, text, HUD, interface")


def resize_wide(image: Image.Image, width: int = 768) -> Image.Image:
    height = round(image.height * width / image.width / 8) * 8
    return image.resize((width, height), Image.Resampling.LANCZOS)


def canny_control(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def panel(input_image: Image.Image, edge: Image.Image,
          outputs: list[tuple[str, Image.Image]]) -> Image.Image:
    items = [("source", input_image), ("Canny", edge)] + outputs
    w, h = input_image.size
    canvas = Image.new("RGB", (w * len(items), h + 38), "#151515")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        canvas.paste(image, (i * w, 38))
        draw.text((i * w + 12, 12), label, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--selection", choices=["primary_test", "validation_scene_coverage"],
                        default="primary_test")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.45, 0.65])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--style-image", type=Path)
    parser.add_argument("--ip-scale", type=float, default=0.6)
    parser.add_argument("--ip-style-only", action="store_true")
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--guidance", type=float, default=7.0)
    args = parser.parse_args()
    if args.sample < 0 or any(not 0 < x <= 1 for x in args.strengths):
        raise ValueError("Invalid sample/strength")
    root = configure_workspace()
    preflight_root = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight_root / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight_root / "controlnet_canny.json").read_text(encoding="utf-8"))
    report_root = (args.lora.parent.parent / f"probe_{args.lora.parent.name}_{args.selection}_sample{args.sample}_{args.tag}"
                   if args.lora else preflight_root)
    index_path = root / "runs/content-quality-eval-20261001/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = index["selections"][args.selection]["paths"]
    if args.sample >= len(selected):
        raise ValueError("Sample index outside selected evaluation group")
    input_path = Path(selected[args.sample])
    source = resize_wide(Image.open(input_path).convert("RGB"), args.width)
    edge = canny_control(source)
    report_root.mkdir(parents=True, exist_ok=True)
    source.save(report_root / "source.png")
    edge.save(report_root / "control_canny.png")

    torch.backends.cuda.matmul.allow_tf32 = False
    network = ControlNetModel.from_pretrained(
        control["snapshot"], variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model["snapshot"], controlnet=network, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if args.lora:
        pipe.load_lora_weights(args.lora.parent, weight_name=args.lora.name, adapter_name="game")
        pipe.set_adapters("game", adapter_weights=args.lora_scale)
    style_reference = None
    if args.style_image:
        adapter_info = json.loads((preflight_root / "ip_adapter_plus.json").read_text(encoding="utf-8"))
        pipe.load_ip_adapter(adapter_info["snapshot"], subfolder="models",
                             weight_name="ip-adapter-plus_sd15.safetensors",
                             local_files_only=True)
        scale = ({"up": {"block_0": [0.0, args.ip_scale, 0.0]}}
                 if args.ip_style_only else args.ip_scale)
        pipe.set_ip_adapter_scale(scale)
        style_reference = Image.open(args.style_image).convert("RGB")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    results = []
    for strength in args.strengths:
        seed = 20261001
        with torch.inference_mode():
            output = pipe(
                prompt=args.prompt, negative_prompt=NEGATIVE,
                image=source, control_image=edge,
                strength=strength, controlnet_conditioning_scale=args.control_scale,
                guidance_scale=args.guidance, num_inference_steps=args.steps,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                ip_adapter_image=style_reference,
            ).images[0]
        name = f"teacher_strength_{strength:.2f}.png"
        output.save(report_root / name)
        results.append((f"teacher strength {strength:.2f}", output))
        write_json(report_root / "probe_status.json", {
            "status": "visual_review_required", "source": str(input_path),
            "selection": args.selection, "sample_index": args.sample,
            "source_sha256": file_hash(input_path), "size": list(source.size),
            "model_revision": model["revision"],
            "controlnet_revision": control["revision"],
            "strengths_completed": [float(x[0].split()[-1]) for x in results],
            "steps": args.steps, "prompt": args.prompt, "negative_prompt": NEGATIVE,
            "lora": str(args.lora) if args.lora else None,
            "lora_scale": args.lora_scale if args.lora else None,
            "style_reference": str(args.style_image) if args.style_image else None,
            "ip_scale": args.ip_scale if args.style_image else None,
            "ip_style_only": args.ip_style_only if args.style_image else None,
            "control_scale": args.control_scale, "guidance": args.guidance,
            "warning": ("Visual acceptance remains required; this is a teacher probe."
                        if args.lora else "The base model is not trained on the project's approved target set."),
        })
        print(name, "done", flush=True)
    panel(source, edge, results).save(report_root / "probe_comparison.jpg", quality=93)
    print("comparison", report_root / "probe_comparison.jpg", flush=True)


if __name__ == "__main__":
    main()
