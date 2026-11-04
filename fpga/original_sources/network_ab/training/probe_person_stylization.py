"""Probe controlled game rendering of people over an already styled background."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                       UniPCMultistepScheduler)
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from .postprocess_teacher_video import model_mask
from .probe_diffusion_teacher import canny_control
from .runtime import configure_workspace, file_hash, write_json


PROMPT = ("a dxgame screenshot, first-person survival video game rendering, "
          "realistic game characters walking in a tree-lined park, faithful pose "
          "and clothes, anatomically correct face, detailed game textures, natural daylight")
NEGATIVE = ("deformed face, distorted face, different person, extra person, duplicate body, "
            "mutated hands, cartoon, anime, painting, blurry, oversmoothed, text, HUD")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.45])
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.strengths):
        raise ValueError("Every strength must be in (0, 1]")
    root = configure_workspace()
    state = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    locked_state = json.loads((args.locked_dir / "review.json").read_text(encoding="utf-8"))
    if any(index not in locked_state["indices"] for index in args.frame_indices):
        raise ValueError("Frame not present in locked result")
    if state["lora_sha256"] != locked_state["lora_sha256"]:
        raise ValueError("LoRA mismatch")
    weights = Path(state["lora_weights"])
    if file_hash(weights) != state["lora_sha256"]:
        raise ValueError("LoRA weights changed")
    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    if file_hash(checkpoint) != locked_state["segmenter_sha256"]:
        raise ValueError("Person segmenter changed")
    segmenter = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    segmenter.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    segmenter.eval()
    torch.set_num_threads(4)

    preflight = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight / "controlnet_canny.json").read_text(encoding="utf-8"))
    network = ControlNetModel.from_pretrained(
        control["snapshot"], variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model["snapshot"], controlnet=network, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(weights.parent, weight_name=weights.name, adapter_name="game")
    pipe.set_adapters("game", adapter_weights=1.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in args.frame_indices:
        source = np.asarray(Image.open(args.input_dir / f"source_{index:06d}.png").convert("RGB"))
        locked = np.asarray(Image.open(args.locked_dir / f"locked_{index:06d}.png").convert("RGB"))
        person = model_mask(segmenter, source)
        hard_person = cv2.dilate((person > 0.25).astype(np.uint8),
                                 np.ones((5, 5), np.uint8), iterations=1)
        mask = Image.fromarray((255 * hard_person).astype(np.uint8), mode="L")
        control_image = canny_control(Image.fromarray(source))
        variants = [source, locked]
        for strength in args.strengths:
            with torch.inference_mode():
                generated = pipe(
                    prompt=PROMPT, negative_prompt=NEGATIVE,
                    image=Image.fromarray(locked), mask_image=mask,
                    control_image=control_image,
                    strength=strength, controlnet_conditioning_scale=0.8,
                    guidance_scale=7.0, num_inference_steps=32,
                    generator=torch.Generator(device="cpu").manual_seed(20261001),
                ).images[0]
            pixels = np.asarray(generated.convert("RGB"))
            result = np.clip(pixels.astype(np.float32) * person[:, :, None] +
                             locked.astype(np.float32) * (1 - person[:, :, None]),
                             0, 255).astype(np.uint8)
            label = f"person_{index:06d}_s{strength:.2f}.png"
            Image.fromarray(result).save(args.output_dir / label)
            variants.append(result)
            records.append({"index": index, "strength": strength,
                            "image": str(args.output_dir / label)})
            print(f"frame {index} strength {strength:.2f}", flush=True)
        panel = Image.new("RGB", (768 * len(variants), 456), "#151515")
        draw = ImageDraw.Draw(panel)
        labels = ["SOURCE", "PEOPLE-LOCKED"] + [f"PERSON {value:.2f}" for value in args.strengths]
        for position, (pixels, label) in enumerate(zip(variants, labels)):
            x = position * 768
            panel.paste(Image.fromarray(pixels), (x, 24))
            draw.text((x + 12, 5), label, fill="white")
        panel.save(args.output_dir / f"review_{index:06d}.jpg", quality=96)

    record = {"schema": "person-stylization-probe-v1", "status": "visual_review_required",
              "indices": args.frame_indices, "strengths": args.strengths,
              "lora_sha256": state["lora_sha256"], "prompt": PROMPT,
              "negative_prompt": NEGATIVE, "variants": records}
    write_json(args.output_dir / "review.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
