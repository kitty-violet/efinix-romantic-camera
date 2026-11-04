"""Test motion-aligned deflickering on an already rendered held-out clip.

This is an image-space diagnostic, not a temporally trained video model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .postprocess_teacher_video import warp_errors
from .runtime import file_hash, write_json


def stabilize(previous_source: np.ndarray, current_source: np.ndarray,
              previous_result: np.ndarray, current_result: np.ndarray,
              alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp the previous output with source motion, rejecting occlusions."""
    height, width = current_source.shape[:2]
    previous_gray = cv2.cvtColor(previous_source, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_source, cv2.COLOR_RGB2GRAY)
    backward = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    mx, my = xx + backward[:, :, 0], yy + backward[:, :, 1]
    warp = lambda pixels: cv2.remap(pixels, mx, my, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
    warped_source = warp(previous_source)
    warped_result = warp(previous_result)
    warped_forward = warp(forward)
    source_error = np.abs(current_source.astype(np.float32) -
                          warped_source.astype(np.float32)).mean(axis=2)
    cycle_error = np.linalg.norm(backward + warped_forward, axis=2)
    in_bounds = (mx >= 0) & (mx < width - 1) & (my >= 0) & (my < height - 1)
    confidence = (np.clip((28.0 - source_error) / 16.0, 0.0, 1.0) *
                  np.clip((3.0 - cycle_error) / 2.0, 0.0, 1.0) *
                  in_bounds.astype(np.float32))
    confidence = cv2.GaussianBlur(confidence, (0, 0), 1.0)
    blend = (alpha * confidence)[:, :, None]
    corrected = (current_result.astype(np.float32) * (1.0 - blend) +
                 warped_result.astype(np.float32) * blend)
    return np.clip(corrected, 0, 255).astype(np.uint8), confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    state = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "rendered_visual_review_required":
        raise ValueError("Source render is incomplete")
    if file_hash(Path(state["video"])) != state["video_sha256"]:
        raise ValueError("Source video changed")
    indices = state["source_frame_indices"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "source_raw_stabilized.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             state["fps"], (2304, 456))
    if not writer.isOpened():
        raise RuntimeError("Could not open output video")
    previous_source = previous_raw = previous_corrected = None
    errors = []
    confidence_fractions = []
    try:
        for position, index in enumerate(indices):
            source = np.asarray(Image.open(args.input_dir /
                                           f"source_{index:06d}.png").convert("RGB"))
            raw = np.asarray(Image.open(args.input_dir /
                                        f"styled_{index:06d}.png").convert("RGB"))
            if previous_source is None:
                corrected = raw.copy()
            else:
                corrected, confidence = stabilize(previous_source, source,
                                                  previous_corrected, raw,
                                                  args.alpha)
                confidence_fractions.append(float((confidence > 0.5).mean()))
                errors.append(warp_errors(previous_source, source,
                                          previous_raw, raw,
                                          previous_corrected, corrected))
            panel = np.zeros((456, 2304, 3), dtype=np.uint8)
            panel[:] = (21, 21, 21)
            panel[24:, 0:768] = source
            panel[24:, 768:1536] = raw
            panel[24:, 1536:2304] = corrected
            for x, label in ((12, "SOURCE"), (780, "FRAMEWISE RAW"),
                             (1548, f"FLOW BLEND {args.alpha:.2f}")):
                cv2.putText(panel, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if position in (0, len(indices) // 2, len(indices) - 1):
                Image.fromarray(panel).save(
                    args.output_dir / f"review_{position:02d}.jpg", quality=94)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            previous_source, previous_raw, previous_corrected = source, raw, corrected
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(target))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count != len(indices):
        raise RuntimeError(f"Video frame count mismatch: {count}")
    result = {
        "schema": "temporal-stabilization-probe-v1",
        "status": "visual_review_required",
        "source_render": str(Path(state["video"])),
        "source_render_sha256": state["video_sha256"],
        "lora_sha256": state["lora_sha256"],
        "alpha": args.alpha,
        "flow": "Farneback forward-backward check with source photometric rejection",
        "frames": count,
        "fps": state["fps"],
        "mean_confident_fraction": float(np.mean(confidence_fractions)),
        "flow_aligned_l1": {
            key: float(np.mean([row[key] for row in errors]))
            for key in ("source_l1", "raw_l1", "corrected_l1")
        },
        "video": str(target),
        "video_sha256": file_hash(target),
        "limitation": "Pixel-space blending can reduce flicker but cannot recover consistent identities or textures invented independently by each diffusion frame.",
    }
    write_json(args.output_dir / "review.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
