"""Try motion-warped prior output as the input of a people-locked inpaint step.

This is a short diagnostic, not a trained video model. The first frame is an
existing independently generated keyframe; subsequent frames are generated.
"""
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

from .postprocess_teacher_video import model_mask, warp_errors
from .probe_diffusion_teacher import canny_control
from .probe_temporal_stabilization import stabilize
from .render_people_locked_inpaint import NEGATIVE, PROMPT
from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--keyframes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--motion-alpha", type=float, default=0.85)
    args = parser.parse_args()
    if not 0 < args.strength <= 1 or not 0 <= args.motion_alpha <= 1:
        raise ValueError("Invalid strength or motion alpha")
    if len(args.frame_indices) < 2 or args.frame_indices != sorted(set(args.frame_indices)):
        raise ValueError("Need at least two distinct ascending frame indices")
    root = configure_workspace()
    state = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    key_state = json.loads((args.keyframes_dir / "review.json").read_text(encoding="utf-8"))
    if any(index not in key_state["indices"] for index in args.frame_indices):
        raise ValueError("Indices must be present in source and keyframe outputs")
    if state["lora_sha256"] != key_state["lora_sha256"]:
        raise ValueError("Keyframes and source use different LoRA weights")
    if file_hash(Path(key_state["video"])) != key_state["video_sha256"]:
        raise ValueError("Keyframe output changed")
    weights = Path(state["lora_weights"])
    if file_hash(weights) != state["lora_sha256"]:
        raise ValueError("LoRA weights changed")

    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    if file_hash(checkpoint) != key_state["segmenter_sha256"]:
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

    first = args.frame_indices[0]
    previous_source = np.asarray(Image.open(args.input_dir / f"source_{first:06d}.png").convert("RGB"))
    previous_result = np.asarray(Image.open(args.keyframes_dir / f"locked_{first:06d}.png").convert("RGB"))
    Image.fromarray(previous_result).save(args.output_dir / f"temporal_{first:06d}.png")
    metrics = []
    for position, index in enumerate(args.frame_indices[1:], 1):
        source = np.asarray(Image.open(args.input_dir / f"source_{index:06d}.png").convert("RGB"))
        independent = np.asarray(Image.open(args.keyframes_dir / f"locked_{index:06d}.png").convert("RGB"))
        person = model_mask(segmenter, source)
        hard_person = cv2.dilate((person > 0.25).astype(np.uint8),
                                 np.ones((5, 5), np.uint8), iterations=1)
        mask = Image.fromarray((255 * (1 - hard_person)).astype(np.uint8), mode="L")
        warm, confidence = stabilize(previous_source, source, previous_result,
                                     source, args.motion_alpha)
        # Keep source identities in the input as well as in the output.
        warm = np.clip(warm.astype(np.float32) * (1 - person[:, :, None]) +
                       source.astype(np.float32) * person[:, :, None],
                       0, 255).astype(np.uint8)
        with torch.inference_mode():
            generated = pipe(
                prompt=PROMPT, negative_prompt=NEGATIVE,
                image=Image.fromarray(warm), mask_image=mask,
                control_image=canny_control(Image.fromarray(source)),
                strength=args.strength, controlnet_conditioning_scale=0.8,
                guidance_scale=7.0, num_inference_steps=32,
                generator=torch.Generator(device="cpu").manual_seed(20261001),
            ).images[0]
        background = np.asarray(generated.convert("RGB"))
        result = np.clip(background.astype(np.float32) * (1 - person[:, :, None]) +
                         source.astype(np.float32) * person[:, :, None],
                         0, 255).astype(np.uint8)
        Image.fromarray(result).save(args.output_dir / f"temporal_{index:06d}.png")
        panel = Image.new("RGB", (2304, 456), "#151515")
        for x, pixels in ((0, source), (768, independent), (1536, result)):
            panel.paste(Image.fromarray(pixels), (x, 24))
        draw = ImageDraw.Draw(panel)
        for x, label in ((12, "SOURCE"), (780, "INDEPENDENT INPAINT"),
                         (1548, "WARM-START INPAINT")):
            draw.text((x, 5), label, fill="white")
        panel.save(args.output_dir / f"review_{index:06d}.jpg", quality=95)
        metric = warp_errors(previous_source, source, previous_result, independent,
                             previous_result, result)
        metrics.append({"index": index, "confidence_gt_0_5": float((confidence > 0.5).mean()),
                        "flow_l1_independent": metric["raw_l1"],
                        "flow_l1_temporal": metric["corrected_l1"]})
        print(f"frame {position + 1}/{len(args.frame_indices)} source={index}", flush=True)
        previous_source, previous_result = source, result

    result = {
        "schema": "temporal-people-inpaint-probe-v1", "status": "visual_review_required",
        "indices": args.frame_indices, "strength": args.strength,
        "motion_alpha": args.motion_alpha, "metrics": metrics,
        "lora_sha256": state["lora_sha256"],
        "keyframe_video_sha256": key_state["video_sha256"],
        "limitation": "Only a sparse-frame probe; warm-starting may preserve hallucinations or reduce style strength.",
    }
    write_json(args.output_dir / "review.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
