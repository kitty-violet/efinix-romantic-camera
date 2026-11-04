"""Protect people/vehicles and damp flicker in an exploratory teacher clip."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.hub import download_url_to_file
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.transforms import functional as TF

from .runtime import configure_workspace, file_hash, write_json


WEIGHT_URL = "https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth"
CLASSES = (2, 6, 7, 14, 15)  # bicycle, bus, car, motorbike, person


def video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer failed")
    try:
        for _ in range(2):
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count != len(frames) * 2:
        raise RuntimeError(f"MP4 count mismatch: {count}")


def model_mask(model, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
    tensor = TF.normalize(TF.to_tensor(Image.fromarray(small)),
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]).unsqueeze(0)
    with torch.inference_mode():
        labels = model(tensor)["out"].argmax(1)[0].numpy().astype(np.uint8)
    mask = np.isin(labels, CLASSES).astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(mask, (0, 0), 2.0).astype(np.float32) / 255.0


def flow_smooth(previous_source: np.ndarray, current_source: np.ndarray,
                previous_result: np.ndarray, current_result: np.ndarray,
                preserve: np.ndarray) -> np.ndarray:
    h, w = current_source.shape[:2]
    current_gray = cv2.cvtColor(current_source, cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(previous_source, cv2.COLOR_RGB2GRAY)
    back = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None,
        pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x, map_y = grid_x + back[:, :, 0], grid_y + back[:, :, 1]
    old_result = cv2.remap(previous_result, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    old_source = cv2.remap(previous_source, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    source_error = np.abs(current_source.astype(np.float32) -
                          old_source.astype(np.float32)).mean(2)
    result_error = np.abs(current_result.astype(np.float32) -
                          old_result.astype(np.float32)).mean(2)
    valid = (source_error < 18) & (result_error < 70)
    alpha = (0.25 * valid.astype(np.float32) * (1.0 - preserve))[:, :, None]
    return np.clip(current_result * (1 - alpha) + old_result * alpha,
                   0, 255).astype(np.uint8)


def warp_errors(previous_source: np.ndarray, current_source: np.ndarray,
                previous_raw: np.ndarray, current_raw: np.ndarray,
                previous_corrected: np.ndarray, current_corrected: np.ndarray) -> dict:
    h, w = current_source.shape[:2]
    backward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(current_source, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(previous_source, cv2.COLOR_RGB2GRAY), None,
        0.5, 3, 21, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = xx + backward[:, :, 0], yy + backward[:, :, 1]
    warp = lambda image: cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
    source_error = np.abs(current_source.astype(np.float32) -
                          warp(previous_source).astype(np.float32)).mean(2)
    stable = source_error < 18
    return {
        "stable_fraction": float(stable.mean()),
        "source_l1": float(source_error[stable].mean()),
        "raw_l1": float(np.abs(current_raw.astype(np.float32) -
                               warp(previous_raw).astype(np.float32)).mean(2)[stable].mean()),
        "corrected_l1": float(np.abs(current_corrected.astype(np.float32) -
                                     warp(previous_corrected).astype(np.float32)).mean(2)[stable].mean()),
    }


def main() -> None:
    root = configure_workspace()
    folder = root / "runs/game-style-lora-20261002-01/temporal-validation-20261002"
    state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "rendered_visual_review_required":
        raise ValueError("Raw teacher video is incomplete")
    weight = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    weight.parent.mkdir(parents=True, exist_ok=True)
    if not weight.is_file():
        download_url_to_file(WEIGHT_URL, str(weight), hash_prefix="d234d4ea", progress=True)
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    model = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    model.load_state_dict(checkpoint)
    model.eval()
    torch.set_num_threads(4)
    indices = state["source_frame_indices"]
    panels = []
    prev_source = prev_result = prev_styled = None
    semantic_fraction = []
    temporal_errors = []
    for i, frame in enumerate(indices):
        source = np.asarray(Image.open(folder / f"source_{frame:06d}.png").convert("RGB"))
        styled = np.asarray(Image.open(folder / f"styled_{frame:06d}.png").convert("RGB"))
        mask = model_mask(model, source)
        semantic_fraction.append(float((mask > .5).mean()))
        Image.fromarray(np.uint8(mask * 255)).save(folder / f"protected_mask_{frame:06d}.png")
        # Keep source geometry for people/vehicles; retain mild generated color.
        protected = (source.astype(np.float32) * .85 + styled.astype(np.float32) * .15)
        composed = np.clip(styled * (1 - mask[:, :, None]) +
                           protected * mask[:, :, None], 0, 255).astype(np.uint8)
        if prev_source is not None:
            composed = flow_smooth(prev_source, source, prev_result, composed, mask)
            temporal_errors.append(warp_errors(
                prev_source, source, prev_styled, styled, prev_result, composed))
        Image.fromarray(composed).save(folder / f"corrected_{frame:06d}.png")
        prev_source, prev_result, prev_styled = source, composed, styled
        panel = Image.new("RGB", (2304, 456), "#151515")
        panel.paste(Image.fromarray(source), (0, 24))
        panel.paste(Image.fromarray(styled), (768, 24))
        panel.paste(Image.fromarray(composed), (1536, 24))
        draw = ImageDraw.Draw(panel)
        for x, label in ((12, "ORIGINAL"), (780, "RAW TEACHER"),
                         (1548, "PROTECTED + FLOW")):
            draw.text((x, 5), label, fill="white")
        if i == 0:
            panel.save(folder / "review_frame_121.jpg", quality=94)
        panels.append(np.asarray(panel))
        print(f"protected {i + 1}/{len(indices)} fraction={semantic_fraction[-1]:.3f}", flush=True)
    target = folder / "source_vs_raw_vs_corrected.mp4"
    video(target, panels)
    write_json(folder / "postprocess_review.json", {
        "status": "rendered_visual_review_required",
        "segmentation_source": WEIGHT_URL, "segmentation_sha256": file_hash(weight),
        "segmentation_classes": list(CLASSES), "protected_fraction": semantic_fraction,
        "flow_aligned_errors": temporal_errors,
        "mean_flow_aligned_l1": {key: float(np.mean([row[key] for row in temporal_errors]))
                                 for key in ("source_l1", "raw_l1", "corrected_l1")},
        "video": str(target), "video_sha256": file_hash(target),
        "limitation": "Foreground source pixels are preserved rather than game-rendered; semantic segmentation misses some objects. Optical flow smoothing is exploratory."})
    print(target, flush=True)


if __name__ == "__main__":
    main()
