"""Fixed target-derived appearance map: 1080 input -> 720 look -> 1080 output.

This is a material/identity/temporal safety control, not a learned game model.
It never invents geometry or textures and uses one fixed map for the full clip.
"""
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
from .render_temporal_native_detail import encode_pipe
from .runtime import configure_workspace, file_hash, write_json


QUANTILES = (0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0)


def video_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (1080, 1920):
                raise ValueError(f"Wrong source size at {index}")
            yield index, frame
            index += 1
        if index != 150:
            raise ValueError(f"Expected 150 unique source frames, got {index}")
    finally:
        capture.release()


def fit_parameters(video: Path, selection: Path) -> dict:
    meta = json.loads((video.parent / "clip.json").read_text(encoding="utf-8"))
    if (meta["status"] != "selected_visual_review" or
            meta["output_sha256"] != file_hash(video) or
            meta["frames"] != 150 or meta["fps"] != 30.0):
        raise ValueError("Selected SDR source clip drift")
    target = json.loads(selection.read_text(encoding="utf-8"))
    if (target["schema"] != "focused-urban-target-selection-v1" or
            target["status"] != "reviewed_focused_urban" or
            target["image_count"] != len(target["records"])):
        raise ValueError("Target selection drift")
    manifest = Path(target["parent_manifest"])
    if file_hash(manifest) != target["parent_manifest_sha256"]:
        raise ValueError("Parent target manifest drift")
    source_labs = []
    for index, frame in video_frames(video):
        if index % 10 == 0:
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            source_labs.append(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3)[::4])
    target_labs = []
    for row in target["records"]:
        path = manifest.parent / row["file"]
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Target changed: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (720, 1280):
            raise ValueError(f"Unreadable target: {path}")
        x1, y1, x2, y2 = row["safe_roi_xyxy"]
        crop = cv2.resize(frame[y1:y2, x1:x2], (320, 180),
                          interpolation=cv2.INTER_AREA)
        target_labs.append(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[::4])
    src = np.concatenate(source_labs).astype(np.float32)
    dst = np.concatenate(target_labs).astype(np.float32)
    src_l = np.percentile(src[:, 0], QUANTILES)
    dst_l = np.percentile(dst[:, 0], QUANTILES)
    # Purely monotone luminance and restrained chroma shifts preserve marble,
    # clothing and edges. Parameters are global across all 150 frames.
    src_l = np.maximum.accumulate(src_l + np.arange(len(src_l)) * 0.001)
    shifts = []
    for threshold in (64, 128, 192, 255):
        src_bin = src[(src[:, 0] >= threshold - 64) & (src[:, 0] < threshold)]
        dst_bin = dst[(dst[:, 0] >= threshold - 64) & (dst[:, 0] < threshold)]
        if len(src_bin) < 100 or len(dst_bin) < 100:
            shift = np.zeros(2, dtype=np.float32)
        else:
            shift = np.clip(np.median(dst_bin[:, 1:], axis=0) -
                            np.median(src_bin[:, 1:], axis=0), -10, 10)
        shifts.append(shift.tolist())
    return {"schema": "material-safe-720-parameters-v1",
            "source_video": str(video), "source_sha256": file_hash(video),
            "target_selection": str(selection),
            "target_selection_sha256": file_hash(selection),
            "fit_source_frames": list(range(0, 150, 10)),
            "quantiles": list(QUANTILES),
            "source_l_quantiles": src_l.tolist(),
            "target_l_quantiles": dst_l.tolist(),
            "ab_shifts_by_l_range": shifts,
            "input_resolution": [1920, 1080],
            "processing_resolution": [1280, 720],
            "output_resolution": [1920, 1080],
            "fps": 30, "frames": 150,
            "color_model": "fixed full-clip Lab quantile map; no per-frame re-estimation",
            "limitation": "Material-safe deterministic appearance control, not a learned game rendering network."}


def process_frame_720(frame: np.ndarray, params: dict, strength: float,
                      person_mask: np.ndarray | None = None,
                      person_weight: float = 0.35) -> np.ndarray:
    if not 0 <= strength <= 1:
        raise ValueError("Strength outside 0..1")
    if not 0 <= person_weight <= 1:
        raise ValueError("Person weight outside 0..1")
    small = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    if person_mask is None:
        person_mask = np.zeros((720, 1280), dtype=np.float32)
    if person_mask.shape != (720, 1280):
        raise ValueError("Person mask must be 1280x720")
    person_mask = np.clip(person_mask.astype(np.float32), 0, 1)
    gain = strength * (1.0 - (1.0 - person_weight) * person_mask)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_l = lab[:, :, 0].copy()
    mapped = np.interp(original_l,
                       params["source_l_quantiles"],
                       params["target_l_quantiles"]).astype(np.float32)
    # Allow a stronger game-like tone on the marble while limiting change on
    # the person. No geometry, texture, tile seam or clothing is generated.
    delta_l = np.minimum(mapped - original_l, 12.0)
    delta_l = np.maximum(delta_l, -(78.0 - 53.0 * person_mask))
    lab[:, :, 0] = np.clip(original_l + gain * delta_l, 0, 255)
    shifts = np.asarray(params["ab_shifts_by_l_range"], dtype=np.float32)
    for channel in range(2):
        chroma_shift = np.interp(original_l, (32, 96, 160, 224),
                                 shifts[:, channel]).astype(np.float32)
        lab[:, :, 1 + channel] = np.clip(
            lab[:, :, 1 + channel] + gain * chroma_shift, 0, 255)
    styled_720 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return styled_720


def process_frame(frame: np.ndarray, params: dict, strength: float,
                  person_mask: np.ndarray | None = None,
                  person_weight: float = 0.35) -> np.ndarray:
    styled_720 = process_frame_720(frame, params, strength,
                                   person_mask, person_weight)
    # The upscale uses the 720p processed image only; no hidden 1080p source
    # detail is fused after the model-resolution stage.
    return cv2.resize(styled_720, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-clip", type=Path, required=True)
    parser.add_argument("--target-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--person-weight", type=float, default=0.35)
    parser.add_argument("--no-person-mask", action="store_true",
                        help="Apply one deterministic map to all pixels; avoids mask-edge jitter")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = fit_parameters(args.source_clip, args.target_selection)
    params["strength"] = args.strength
    params["person_weight"] = args.person_weight
    segmenter = None
    if not args.no_person_mask:
        root = configure_workspace()
        checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        segmenter = lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=21)
        segmenter.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        segmenter.eval()
        torch.set_num_threads(4)
        params["person_segmenter_sha256"] = file_hash(checkpoint)
    params["person_mask_policy"] = "none_fixed_pixel_map" if args.no_person_mask else "framewise_lraspp"
    write_json(args.output_dir / "parameters.json", params)
    selected = (0, 75, 149)
    encoder = None
    video = args.output_dir / "material_safe_1080p30_5s.mp4"
    if not args.preview:
        encoder = encode_pipe(video, 1920, 1080, 30)
    frame_count = 0
    try:
        for index, frame in video_frames(args.source_clip):
            if args.preview and index not in selected:
                frame_count += 1
                continue
            if segmenter is None:
                person_mask = None
            else:
                small_rgb = cv2.cvtColor(cv2.resize(frame, (1280, 720),
                                                    interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2RGB)
                person_mask = model_mask(segmenter, small_rgb)
            result = process_frame(frame, params, args.strength,
                                   person_mask, args.person_weight)
            if index in selected:
                small_source = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                small_result = cv2.resize(result, (960, 540), interpolation=cv2.INTER_AREA)
                panel = np.concatenate((small_source, small_result), axis=1)
                cv2.imwrite(str(args.output_dir / f"review_{index:03d}.jpg"), panel,
                            [cv2.IMWRITE_JPEG_QUALITY, 96])
            if encoder is not None:
                if encoder.stdin is None:
                    raise RuntimeError("Encoder stdin missing")
                encoder.stdin.write(cv2.cvtColor(result, cv2.COLOR_BGR2RGB).tobytes())
            frame_count += 1
    finally:
        if encoder is not None and encoder.stdin is not None:
            encoder.stdin.close()
    if encoder is not None:
        errors = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        if encoder.wait(timeout=120):
            raise RuntimeError(f"Encoding failed: {errors[-1000:]}")
        capture = cv2.VideoCapture(str(video))
        count = 0
        while capture.read()[0]:
            count += 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if (count, width, height) != (150, 1920, 1080) or abs(fps - 30) > 0.1:
            raise RuntimeError(f"Output drift: {count} frames {width}x{height} {fps}fps")
    record = {"schema": "material-safe-720-render-v1",
              "status": "preview_only" if args.preview else "rendered_visual_review_required",
              "parameters": str(args.output_dir / "parameters.json"),
              "parameters_sha256": file_hash(args.output_dir / "parameters.json"),
              "source": str(args.source_clip), "source_sha256": params["source_sha256"],
              "frames": frame_count, "fps": 30, "strength": args.strength,
              "person_weight": args.person_weight,
              "person_mask_policy": params["person_mask_policy"],
              "processing_resolution": [1280, 720],
              "output_resolution": [1920, 1080],
              "video": str(video) if encoder is not None else None,
              "video_sha256": file_hash(video) if encoder is not None else None,
              "limitation": params["limitation"]}
    write_json(args.output_dir / "status.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
