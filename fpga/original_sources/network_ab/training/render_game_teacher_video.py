"""Render an honest short consecutive-frame validation clip from held-out video."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       UniPCMultistepScheduler)

from .media import decode_frames
from .probe_diffusion_teacher import canny_control, resize_wide
from .runtime import configure_workspace, file_hash, write_json


PROMPT = ("a dxgame screenshot of a tree-lined urban road with parked motorcycles, "
          "photorealistic first-person survival video game rendering, detailed "
          "realistic road and vehicle textures, natural daylight")
NEGATIVE = "cartoon, anime, painting, blurry, oversmoothed, text, HUD, interface"


def write_video(path: Path, frames: list[Image.Image], fps: int) -> None:
    w, h = frames[0].size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create validation video: {path}")
    try:
        for _ in range(2):
            for image in frames:
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count != len(frames) * 2:
        raise RuntimeError(f"Video frame count mismatch: {count}")


def main() -> None:
    root = configure_workspace()
    run = root / "runs/game-style-lora-20261002-01"
    out = run / "temporal-validation-20261002"
    out.mkdir(parents=True, exist_ok=True)
    preflight = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight / "controlnet_canny.json").read_text(encoding="utf-8"))
    adapter = json.loads((preflight / "ip_adapter_plus.json").read_text(encoding="utf-8"))
    content_manifest = root / "data/processed/content-20261001-v4-minus1ev-draft/manifest.approved.json"
    content = json.loads(content_manifest.read_text(encoding="utf-8"))
    anchor = next(row for row in content["images"] if
                  row["path"].endswith("5ff31bb756cc826b_00000121.png"))
    source_video = Path(anchor["source"])
    if file_hash(source_video) != anchor["source_sha256"]:
        raise ValueError("Held-out source video changed")
    style = root / "data/processed/style-target-20261001-v14-expanded-reviewed/N_18-59-02_00700.png"
    weights = run / "weights_001000/pytorch_lora_weights.safetensors"
    params = {"schema": "game-teacher-consecutive-video-v1",
              "source_video": str(source_video), "source_sha256": file_hash(source_video),
              "source_frame_indices": list(range(121, 169, 3)), "fps": 10,
              "target_reference": str(style), "target_sha256": file_hash(style),
              "lora_weights": str(weights), "lora_sha256": file_hash(weights),
              "model_revision": model["revision"], "control_revision": control["revision"],
              "ip_adapter_revision": adapter["revision"], "prompt": PROMPT,
              "strength": 0.75, "steps": 32, "ip_scale": 0.7,
              "control_scale": 1.2, "guidance": 8.0, "seed": 20261001,
              "warning": "Validation sample, not accepted as faithful game stylization."}
    write_json(out / "config.json", params)
    prepared = []
    for frame_index, pts, array in decode_frames(
            source_video, indices=params["source_frame_indices"], hlg_exposure_stops=-1.0):
        source = ImageOps.fit(Image.fromarray(array), (1280, 720),
                              method=Image.Resampling.BICUBIC)
        if frame_index == 121 and not np.array_equal(
                np.asarray(source), np.asarray(Image.open(anchor["path"]).convert("RGB"))):
            raise RuntimeError("New video decode differs from approved held-out test frame")
        source = resize_wide(source, 768)
        source.save(out / f"source_{frame_index:06d}.png")
        prepared.append((frame_index, pts, source))
    if len(prepared) != len(params["source_frame_indices"]):
        raise RuntimeError("Source decode did not return every requested frame")

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
    pipe.load_ip_adapter(adapter["snapshot"], subfolder="models",
                         weight_name="ip-adapter-plus_sd15.safetensors",
                         local_files_only=True)
    pipe.set_ip_adapter_scale(params["ip_scale"])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    style_image = Image.open(style).convert("RGB")
    panels = []
    for i, (frame_index, pts, source) in enumerate(prepared):
        output_path = out / f"styled_{frame_index:06d}.png"
        if output_path.is_file():
            output = Image.open(output_path).convert("RGB")
        else:
            with torch.inference_mode():
                output = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
                              image=source, control_image=canny_control(source),
                              ip_adapter_image=style_image, strength=params["strength"],
                              controlnet_conditioning_scale=params["control_scale"],
                              guidance_scale=params["guidance"],
                              num_inference_steps=params["steps"],
                              generator=torch.Generator(device="cpu").manual_seed(params["seed"]),
                              output_type="pil").images[0]
            output.save(output_path)
        panel = Image.new("RGB", (1536, 456), "#151515")
        panel.paste(source, (0, 24))
        panel.paste(output, (768, 24))
        ImageDraw.Draw(panel).text((12, 5), f"source frame {frame_index} ({pts:.3f}s)  |  teacher result", fill="white")
        panels.append(panel)
        write_json(out / "status.json", {
            **params, "status": "rendering", "frames_done": i + 1,
            "frames_total": len(prepared)})
        print(f"frame {i + 1}/{len(prepared)} {frame_index}", flush=True)
    video = out / "source_vs_teacher.mp4"
    write_video(video, panels, params["fps"])
    write_json(out / "status.json", {
        **params, "status": "rendered_visual_review_required",
        "frames_done": len(prepared), "video": str(video),
        "video_sha256": file_hash(video)})
    print(video, flush=True)


if __name__ == "__main__":
    main()
