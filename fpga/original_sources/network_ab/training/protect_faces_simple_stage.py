"""Keep real faces and people recognizable in a game-style video probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from .postprocess_teacher_video import model_mask, warp_errors
from .probe_temporal_stabilization import stabilize
from .runtime import configure_workspace, file_hash, write_json


def detected_face_mask(source: np.ndarray, person_mask: np.ndarray,
                       detector: cv2.CascadeClassifier) -> tuple[np.ndarray, list[list[int]]]:
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    candidates = detector.detectMultiScale(gray, scaleFactor=1.05,
                                           minNeighbors=5, minSize=(25, 25))
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    accepted = []
    for x, y, width, height in candidates:
        x, y, width, height = map(int, (x, y, width, height))
        region = person_mask[y:y + height, x:x + width]
        if region.size == 0 or float((region > 0.5).mean()) < 0.65:
            continue
        if y < source.shape[0] * 0.18:
            continue
        center = (x + width // 2, y + height // 2)
        axes = (max(1, round(width * 0.62)), max(1, round(height * 0.68)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        accepted.append([x, y, width, height])
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                            (0, 0), 2.0), accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--background-flow-alpha", type=float, default=0.6)
    parser.add_argument("--person-source-fraction", type=float, default=0.8)
    parser.add_argument("--face-source-fraction", type=float, default=0.98)
    args = parser.parse_args()
    for value in (args.background_flow_alpha, args.person_source_fraction,
                  args.face_source_fraction):
        if not 0 <= value <= 1:
            raise ValueError("Blend fractions must be in [0, 1]")
    root = configure_workspace()
    state = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "rendered_visual_review_required":
        raise ValueError("The framewise render is incomplete")
    if file_hash(Path(state["video"])) != state["video_sha256"]:
        raise ValueError("The framewise video changed")
    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Approved local person segmenter is not cached")
    model = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV face detector is unavailable")
    indices = state["source_frame_indices"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "source_raw_face_protected.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             state["fps"], (2304, 456))
    if not writer.isOpened():
        raise RuntimeError("Could not create face-protection video")
    previous_source = previous_background = previous_raw = previous_final = None
    face_rows = []
    errors = []
    person_fractions = []
    try:
        for position, index in enumerate(indices):
            source = np.asarray(Image.open(args.input_dir /
                                           f"source_{index:06d}.png").convert("RGB"))
            raw = np.asarray(Image.open(args.input_dir /
                                        f"styled_{index:06d}.png").convert("RGB"))
            if previous_source is None:
                background = raw.copy()
            else:
                background, _ = stabilize(previous_source, source,
                                          previous_background, raw,
                                          args.background_flow_alpha)
            person = model_mask(model, source)
            face, boxes = detected_face_mask(source, person, detector)
            person_fractions.append(float((person > 0.5).mean()))
            protected_person = (source.astype(np.float32) * args.person_source_fraction +
                                raw.astype(np.float32) * (1 - args.person_source_fraction))
            final = (background.astype(np.float32) * (1 - person[:, :, None]) +
                     protected_person * person[:, :, None])
            protected_face = (source.astype(np.float32) * args.face_source_fraction +
                              raw.astype(np.float32) * (1 - args.face_source_fraction))
            final = np.clip(final * (1 - face[:, :, None]) +
                            protected_face * face[:, :, None], 0, 255).astype(np.uint8)
            if previous_source is not None:
                errors.append(warp_errors(previous_source, source,
                                          previous_raw, raw,
                                          previous_final, final))
            face_rows.append({"index": index, "boxes": boxes})
            panel = np.zeros((456, 2304, 3), dtype=np.uint8)
            panel[:] = 21
            panel[24:, :768] = source
            panel[24:, 768:1536] = raw
            panel[24:, 1536:] = final
            for x, label in ((12, "SOURCE"), (780, "FRAMEWISE RAW"),
                             (1548, "FACE/PEOPLE PROTECTED")):
                cv2.putText(panel, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if position in (0, len(indices) // 2, len(indices) - 2,
                            len(indices) - 1):
                Image.fromarray(panel).save(
                    args.output_dir / f"review_{position:02d}.jpg", quality=95)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            previous_source, previous_background = source, background
            previous_raw, previous_final = raw, final
            print(f"frame {position + 1}/{len(indices)} faces={len(boxes)}", flush=True)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output))
    encoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = sum(capture.read()[0] for _ in range(encoded))
    capture.release()
    if encoded != len(indices) or decoded != encoded:
        raise RuntimeError("Face-protection video did not decode fully")
    record = {
        "schema": "face-protected-video-probe-v1",
        "status": "visual_review_required",
        "source_video": state["video"],
        "source_video_sha256": state["video_sha256"],
        "lora_sha256": state["lora_sha256"],
        "segmenter_sha256": file_hash(checkpoint),
        "face_detector": "OpenCV haarcascade_frontalface_default.xml",
        "parameters": {
            "background_flow_alpha": args.background_flow_alpha,
            "person_source_fraction": args.person_source_fraction,
            "face_source_fraction": args.face_source_fraction,
        },
        "face_boxes": face_rows,
        "mean_person_fraction": float(np.mean(person_fractions)),
        "flow_aligned_l1": {key: float(np.mean([row[key] for row in errors]))
                            for key in ("source_l1", "raw_l1", "corrected_l1")},
        "video": str(output), "video_sha256": file_hash(output),
        "frames": encoded, "fps": state["fps"],
        "limitation": "People and detected faces use mostly original pixels; this restores geometry but does not create a game-rendered identity. Segmentation boundaries need visual review.",
    }
    write_json(args.output_dir / "review.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
