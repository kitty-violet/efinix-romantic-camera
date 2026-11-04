"""Compare instruction-guided game editing on a fixed held-out content frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from .probe_diffusion_teacher import resize_wide
from .runtime import configure_workspace, file_hash, write_json


INSTRUCTION = (
    "Transform this real street photograph into a photorealistic first-person "
    "3D survival video game screenshot. Render the existing road, trees, "
    "motorcycles and buildings as detailed weathered game assets while keeping "
    "their positions and shapes. Use realistic rough materials and warm daylight."
)
NEGATIVE = "cartoon, anime, oil painting, watercolor, blurry, extra objects, deformed vehicles, text, HUD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--pairs", nargs="+", default=["7.5:1.5", "10:1.2"])
    args = parser.parse_args()
    root = configure_workspace()
    preflight = root / "runs/diffusion-teacher-preflight"
    report = json.loads((preflight / "instruct_pix2pix.json").read_text(encoding="utf-8"))
    index = json.loads((root / "runs/content-quality-eval-20261001/index.json").read_text(encoding="utf-8"))
    source_path = Path(index["selections"]["primary_test"]["paths"][args.sample])
    source = resize_wide(Image.open(source_path).convert("RGB"), args.width)
    target_ref = root / "data/processed/style-target-20261001-v14-expanded-reviewed/N_18-59-02_00700.png"
    reference = resize_wide(Image.open(target_ref).convert("RGB"), args.width)
    out = root / "runs" / (("instruct-pix2pix-lora-probe-20261002-"
                            if args.lora else "instruct-pix2pix-base-probe-20261002-") + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        report["snapshot"], variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, safety_checker=None, requires_safety_checker=False,
        local_files_only=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if args.lora:
        pipe.load_lora_weights(args.lora.parent, weight_name=args.lora.name, adapter_name="game")
        pipe.set_adapters("game", adapter_weights=1.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    conditions = [tuple(map(float, item.split(":"))) for item in args.pairs]
    if any(len(pair) != 2 or pair[0] <= 1 or pair[1] < 1 for pair in conditions):
        raise ValueError("Expected text:image pairs with text > 1 and image >= 1")
    results = []
    for text_scale, image_scale in conditions:
        with torch.inference_mode():
            generated = pipe(
                prompt=INSTRUCTION, negative_prompt=NEGATIVE,
                image=source, num_inference_steps=args.steps,
                guidance_scale=text_scale, image_guidance_scale=image_scale,
                generator=torch.Generator(device="cpu").manual_seed(20261001),
            ).images[0]
        name = f"edit_t{text_scale:g}_i{image_scale:g}.png"
        generated.save(out / name)
        results.append((f"edit text {text_scale:g} image {image_scale:g}", generated))
        print(name, flush=True)
    w, h = source.size
    columns = [("source", source), ("game target reference", reference)] + results
    panel = Image.new("RGB", (w * len(columns), h + 32), "#151515")
    draw = ImageDraw.Draw(panel)
    for i, (label, image) in enumerate(columns):
        panel.paste(image, (i * w, 32))
        draw.text((i * w + 12, 9), label, fill="white")
    panel.save(out / "comparison.jpg", quality=93)
    write_json(out / "review.json", {
        "status": "visual_review_required",
        "source": str(source_path), "source_sha256": file_hash(source_path),
        "target_reference": str(target_ref), "target_sha256": file_hash(target_ref),
        "model_revision": report["revision"], "lora": str(args.lora) if args.lora else None,
        "instruction": INSTRUCTION, "negative_prompt": NEGATIVE, "steps": args.steps,
        "conditions": conditions, "comparison": str(out / "comparison.jpg")})
    print(out / "comparison.jpg", flush=True)


if __name__ == "__main__":
    main()
