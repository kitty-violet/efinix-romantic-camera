"""Probe background-only diffusion while preserving people in the source video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                       UniPCMultistepScheduler)
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from .postprocess_teacher_video import model_mask, warp_errors
from .probe_diffusion_teacher import canny_control
from .runtime import configure_workspace, file_hash, write_json


PROMPT = ("a dxgame screenshot of a tree-lined park walkway, "
          "first-person survival video game rendering, detailed foliage "
          "and path textures, natural daylight")
NEGATIVE = ("extra people, duplicate people, malformed face, malformed body, "
            "cartoon, anime, painting, blurry, oversmoothed, text, HUD")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-indices", nargs="*", type=int)
    parser.add_argument("--strength", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 < args.strength <= 1:
        raise ValueError("strength must be in (0, 1]")
    root = configure_workspace()
    state = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "rendered_visual_review_required":
        raise ValueError("Framewise video is incomplete")
    if file_hash(Path(state["video"])) != state["video_sha256"]:
        raise ValueError("Framewise video changed")
    indices = args.frame_indices or state["source_frame_indices"]
    if any(i not in state["source_frame_indices"] for i in indices):
        raise ValueError("Requested a frame outside the held-out video")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    segmenter = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    segmenter.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    segmenter.eval()
    torch.set_num_threads(4)
    preflight = root / "runs/diffusion-teacher-preflight"
    model = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    control = json.loads((preflight / "controlnet_canny.json").read_text(encoding="utf-8"))
    weights = Path(state["lora_weights"])
    if file_hash(weights) != state["lora_sha256"]:
        raise ValueError("LoRA weights changed")
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
    previous_source = previous_raw = previous_result = None
    errors = []
    person_fractions = []
    for position, index in enumerate(indices):
        source = np.asarray(Image.open(args.input_dir /
                                       f"source_{index:06d}.png").convert("RGB"))
        raw = np.asarray(Image.open(args.input_dir /
                                    f"styled_{index:06d}.png").convert("RGB"))
        person = model_mask(segmenter, source)
        hard_person = cv2.dilate((person > 0.25).astype(np.uint8),
                                 np.ones((5, 5), np.uint8), iterations=1)
        mask = Image.fromarray((255 * (1 - hard_person)).astype(np.uint8), mode="L")
        with torch.inference_mode():
            generated = pipe(
                prompt=PROMPT, negative_prompt=NEGATIVE,
                image=Image.fromarray(source), mask_image=mask,
                control_image=canny_control(Image.fromarray(source)),
                strength=args.strength, controlnet_conditioning_scale=0.8,
                guidance_scale=7.0, num_inference_steps=32,
                generator=torch.Generator(device="cpu").manual_seed(20261001),
            ).images[0]
        background = np.asarray(generated.convert("RGB"))
        # The 4-channel inpaint pipeline preserves unmasked latents. Restore
        # source pixels inside the soft person mask to retain small face detail.
        result = np.clip(background.astype(np.float32) * (1 - person[:, :, None]) +
                         source.astype(np.float32) * person[:, :, None],
                         0, 255).astype(np.uint8)
        Image.fromarray(result).save(args.output_dir / f"locked_{index:06d}.png")
        person_fractions.append(float((person > 0.5).mean()))
        if previous_source is not None:
            errors.append(warp_errors(previous_source, source,
                                      previous_raw, raw,
                                      previous_result, result))
        panel = Image.new("RGB", (2304, 456), "#151515")
        panel.paste(Image.fromarray(source), (0, 24))
        panel.paste(Image.fromarray(raw), (768, 24))
        panel.paste(Image.fromarray(result), (1536, 24))
        import PIL.ImageDraw
        draw = PIL.ImageDraw.Draw(panel)
        for x, label in ((12, "SOURCE"), (780, "FRAMEWISE RAW"),
                         (1548, "PEOPLE-LOCKED INPAINT")):
            draw.text((x, 5), label, fill="white")
        panel.save(args.output_dir / f"review_{index:06d}.jpg", quality=95)
        previous_source, previous_raw, previous_result = source, raw, result
        print(f"frame {position + 1}/{len(indices)} source={index}", flush=True)
    video = args.output_dir / "source_raw_people_locked.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             state["fps"], (2304, 456))
    if not writer.isOpened():
        raise RuntimeError("Could not create people-locked video")
    try:
        for index in indices:
            frame = cv2.imread(str(args.output_dir / f"review_{index:06d}.jpg"))
            if frame is None:
                raise RuntimeError(f"Missing comparison frame {index}")
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    encoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = sum(capture.read()[0] for _ in range(encoded))
    capture.release()
    if encoded != len(indices) or decoded != encoded:
        raise RuntimeError("People-locked video did not decode fully")
    record = {
        "schema": "people-locked-inpaint-probe-v1",
        "status": "visual_review_required",
        "source_video": state["video"],
        "source_video_sha256": state["video_sha256"],
        "lora_weights": str(weights), "lora_sha256": file_hash(weights),
        "segmenter_sha256": file_hash(checkpoint),
        "indices": indices, "person_fractions": person_fractions,
        "strength": args.strength,
        "video": str(video), "video_sha256": file_hash(video),
        "encoded_frames": encoded, "fps": state["fps"],
        "prompt": PROMPT, "negative_prompt": NEGATIVE,
        "flow_aligned_l1": ({key: float(np.mean([row[key] for row in errors]))
                             for key in ("source_l1", "raw_l1", "corrected_l1")}
                            if errors else None),
        "limitation": "People are source pixels, not game-rendered people; no time-aware sampling is used for the background.",
    }
    write_json(args.output_dir / "review.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
