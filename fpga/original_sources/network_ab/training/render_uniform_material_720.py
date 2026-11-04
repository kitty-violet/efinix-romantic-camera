"""Render a scene-independent material simplification trial at 720p.

This deterministic control tests the user's material-scale hypothesis. It is
not a trained style model or evidence of cross-scene generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .render_material_safe_720 import fit_parameters, process_frame_720, video_frames
from .render_temporal_native_detail import encode_pipe
from .runtime import file_hash, write_json


def simplify_materials(frame_720: np.ndarray, amount: float) -> np.ndarray:
    """Reduce tiny intra-surface variation while retaining strong boundaries.

    All coefficients remain fixed throughout the clip. The two bilateral
    scales suppress camera noise and small surface speckles; they do not add
    texture or infer a material class. A source contribution retains details
    on fabric, hair and tile seams.
    """
    if frame_720.shape != (720, 1280, 3):
        raise ValueError("Expected BGR 1280x720 processing frame")
    if not 0 <= amount <= 1:
        raise ValueError("Amount outside 0..1")
    if amount == 0:
        return frame_720.copy()
    # 720p-only filtering: there is no 1080p source-detail shortcut afterward.
    fine = cv2.bilateralFilter(frame_720, 7, 24, 2.5)
    broad = cv2.bilateralFilter(fine, 9, 26, 4.0)
    return cv2.addWeighted(frame_720, 1.0 - amount, broad, amount, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-clip", type=Path, required=True)
    parser.add_argument("--target-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tone-strength", type=float, default=1.0)
    parser.add_argument("--texture-amount", type=float, default=0.8)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.tone_strength <= 1:
        raise ValueError("Tone strength outside 0..1")
    if not 0 <= args.texture_amount <= 1:
        raise ValueError("Texture amount outside 0..1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = fit_parameters(args.source_clip, args.target_selection)
    params.update({"tone_strength": args.tone_strength,
                   "texture_amount": args.texture_amount,
                   "person_mask_policy": "none",
                   "material_operation": "two fixed bilateral scales with source-detail blend",
                   "temporal_policy": "no per-frame fitted values, masks, noise, or generation",
                   "limitation": "Deterministic material-scale test, not a trained game style model."})
    write_json(args.output_dir / "parameters.json", params)
    selected = {0, 75, 149}
    video = args.output_dir / "uniform_material_1080p30_5s.mp4"
    encoder = None if args.preview else encode_pipe(video, 1920, 1080, 30)
    count = 0
    try:
        for index, frame in video_frames(args.source_clip):
            count += 1
            if args.preview and index not in selected:
                continue
            toned = process_frame_720(frame, params, args.tone_strength,
                                      person_weight=1.0)
            simplified = simplify_materials(toned, args.texture_amount)
            result = cv2.resize(simplified, (1920, 1080),
                                interpolation=cv2.INTER_LANCZOS4)
            if index in selected:
                left = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                right = cv2.resize(result, (960, 540), interpolation=cv2.INTER_AREA)
                panel = np.concatenate((left, right), axis=1)
                cv2.imwrite(str(args.output_dir / f"review_{index:03d}.jpg"),
                            panel, [cv2.IMWRITE_JPEG_QUALITY, 96])
                cv2.imwrite(str(args.output_dir / f"toned_{index:03d}.jpg"),
                            toned, [cv2.IMWRITE_JPEG_QUALITY, 96])
                cv2.imwrite(str(args.output_dir / f"simplified_{index:03d}.jpg"),
                            simplified, [cv2.IMWRITE_JPEG_QUALITY, 96])
            if encoder is not None:
                if encoder.stdin is None:
                    raise RuntimeError("Encoder stdin missing")
                encoder.stdin.write(cv2.cvtColor(result, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        if encoder is not None and encoder.stdin is not None:
            encoder.stdin.close()
    if encoder is not None:
        errors = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        if encoder.wait(timeout=120):
            raise RuntimeError(f"Encoding failed: {errors[-1000:]}")
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        metadata = (cap.get(cv2.CAP_PROP_FPS),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if decoded != 150 or metadata != (30.0, 1920, 1080):
            raise RuntimeError(f"Output mismatch: {decoded} frames; {metadata}")
    record = {"schema": "uniform-material-trial-v1",
              "status": "preview_only" if args.preview else "visual_review_required",
              "source": str(args.source_clip),
              "source_sha256": params["source_sha256"],
              "parameters": str(args.output_dir / "parameters.json"),
              "parameters_sha256": file_hash(args.output_dir / "parameters.json"),
              "processing_resolution": [1280, 720],
              "output_resolution": [1920, 1080],
              "fps": 30, "frames": count,
              "video": str(video) if encoder is not None else None,
              "video_sha256": file_hash(video) if encoder is not None else None,
              "limitation": params["limitation"]}
    write_json(args.output_dir / "status.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
