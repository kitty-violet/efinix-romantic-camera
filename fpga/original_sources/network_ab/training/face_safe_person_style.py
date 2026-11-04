"""Preserve source head geometry while matching a rendered person's palette."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from .postprocess_teacher_video import model_mask
from .runtime import configure_workspace, file_hash, write_json


def _palette_match_head(source: np.ndarray, rendered: np.ndarray,
                        box: tuple[int, int, int, int]) -> np.ndarray:
    """Transfer restrained Lab statistics without moving any source features."""
    x, y, width, height = box
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(rendered, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_patch = src_lab[y:y + height, x:x + width].reshape(-1, 3)
    dst_patch = dst_lab[y:y + height, x:x + width].reshape(-1, 3)
    src_mean, dst_mean = src_patch.mean(axis=0), dst_patch.mean(axis=0)
    src_std = np.maximum(src_patch.std(axis=0), 5.0)
    dst_std = np.maximum(dst_patch.std(axis=0), 5.0)
    scale = np.clip(dst_std / src_std, 0.75, 1.25)
    scale = 0.75 + 0.25 * scale
    shift = 0.60 * (dst_mean - src_mean)
    shift[0] = np.clip(shift[0], -12, 12)
    shift[1:] = np.clip(shift[1:], -8, 8)
    graded = np.clip((src_lab - src_mean) * scale + src_mean + shift,
                     0, 255).astype(np.uint8)
    return cv2.cvtColor(graded, cv2.COLOR_LAB2RGB)


def _palette_match_masked(source: np.ndarray, rendered: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(rendered, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_pixels, dst_pixels = src_lab[mask], dst_lab[mask]
    src_mean, dst_mean = src_pixels.mean(axis=0), dst_pixels.mean(axis=0)
    scale = np.clip(np.maximum(dst_pixels.std(axis=0), 5.0) /
                    np.maximum(src_pixels.std(axis=0), 5.0), 0.75, 1.25)
    scale = 0.75 + 0.25 * scale
    shift = 0.60 * (dst_mean - src_mean)
    shift[0] = np.clip(shift[0], -12, 12)
    shift[1:] = np.clip(shift[1:], -8, 8)
    graded = np.clip((src_lab - src_mean) * scale + src_mean + shift,
                     0, 255).astype(np.uint8)
    return cv2.cvtColor(graded, cv2.COLOR_LAB2RGB)


def protect_faces(source: np.ndarray, rendered: np.ndarray, person: np.ndarray,
                  detector: cv2.CascadeClassifier) -> tuple[np.ndarray, list[dict]]:
    """Use rendered bodies and geometry-preserving source head detail."""
    if source.shape != rendered.shape or source.shape[:2] != person.shape:
        raise ValueError("Source, rendered, and person mask sizes differ")
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.08,
                                      minNeighbors=5, minSize=(20, 20))
    result = rendered.astype(np.float32).copy()
    accepted = []
    height, width = person.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (person > 0.5).astype(np.uint8), connectivity=8)
    yy = np.arange(height, dtype=np.float32)[:, None]
    for component in range(1, count):
        x, y, bw, bh, area = map(int, stats[component])
        if area < 1500 or bh < 80:
            continue
        region = labels == component
        head_pixels = region & (yy < y + 0.25 * bh)
        if int(head_pixels.sum()) < 100:
            continue
        graded = _palette_match_masked(source, rendered, head_pixels)
        fade = np.clip((y + 0.28 * bh - yy) / (0.12 * bh), 0.0, 1.0)
        shape = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2.0)
        alpha = (shape * fade * person)[:, :, None]
        result = result * (1.0 - alpha) + graded.astype(np.float32) * alpha
    for box in boxes:
        x, y, bw, bh = map(int, box)
        occupancy = float(person[y:y + bh, x:x + bw].mean())
        if occupancy < 0.70:
            continue
        graded = _palette_match_head(source, rendered, (x, y, bw, bh))
        oval = np.zeros((height, width), dtype=np.float32)
        center = (x + bw // 2, int(y + 0.3 * bh))
        axes = (max(1, int(bw * 0.80)), max(1, int(bh * 0.85)))
        cv2.ellipse(oval, center, axes, 0, 0, 360, 1.0, -1)
        oval = cv2.GaussianBlur(oval, (0, 0), 3.0)
        alpha = (oval * person)[:, :, None]
        result = result * (1.0 - alpha) + graded.astype(np.float32) * alpha
        accepted.append({"box": [x, y, bw, bh], "person_occupancy": occupancy})
    return np.clip(result, 0, 255).astype(np.uint8), accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--person-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()
    root = configure_workspace()
    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    segmenter = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    segmenter.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    segmenter.eval()
    torch.set_num_threads(4)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV face cascade unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in args.frame_indices:
        source = np.asarray(Image.open(args.input_dir / f"source_{index:06d}.png").convert("RGB"))
        rendered = np.asarray(Image.open(args.person_dir /
                                         f"person_{index:06d}_s{args.strength:.2f}.png").convert("RGB"))
        person = model_mask(segmenter, source)
        result, faces = protect_faces(source, rendered, person, detector)
        target = args.output_dir / f"face_safe_{index:06d}.png"
        Image.fromarray(result).save(target)
        panel = Image.new("RGB", (2304, 456), "#151515")
        draw = ImageDraw.Draw(panel)
        for position, (pixels, label) in enumerate(
            ((source, "SOURCE"), (rendered, "PERSON DIFFUSION"),
             (result, "FACE-SAFE PERSON"))
        ):
            x = position * 768
            panel.paste(Image.fromarray(pixels), (x, 24))
            draw.text((x + 12, 5), label, fill="white")
        panel.save(args.output_dir / f"review_{index:06d}.jpg", quality=96)
        records.append({"index": index, "faces": faces, "image": str(target)})
        print(f"frame {index}: {len(faces)} accepted faces", flush=True)
    record = {"schema": "face-safe-person-style-v1", "status": "visual_review_required",
              "strength": args.strength, "segmenter_sha256": file_hash(checkpoint),
              "records": records}
    write_json(args.output_dir / "review.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
