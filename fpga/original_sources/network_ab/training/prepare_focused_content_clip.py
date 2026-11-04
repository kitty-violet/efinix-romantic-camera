"""Extract an SDR 1080p, frame-unique content clip for focused style review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from .media import decode_frames, probe_video
from .render_temporal_native_detail import encode_pipe
from .runtime import file_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--start-seconds", type=float, required=True)
    parser.add_argument("--duration-seconds", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start_seconds < 0 or not 5 <= args.duration_seconds <= 10:
        raise ValueError("Need a nonnegative start and a 5–10 second clip")
    if file_hash(args.source) != args.source_sha256:
        raise ValueError("Source video changed")
    probe = probe_video(args.source)
    if (probe["width"], probe["height"]) != (1920, 1080):
        raise ValueError("Expected landscape 1080p source")
    start = round(args.start_seconds * probe["fps"])
    count = args.duration_seconds * 30
    indices = list(range(start, start + count))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "focused_source_1080p.mp4"
    encoder = encode_pipe(target, 1920, 1080, 30)
    pts_seen = []
    try:
        for number, (index, pts, pixels) in enumerate(decode_frames(
                args.source, indices=indices, hlg_exposure_stops=-1.0)):
            if encoder.stdin is None:
                raise RuntimeError("Encoder stdin missing")
            encoder.stdin.write(pixels.tobytes())
            pts_seen.append(pts)
            if number in (0, count // 2, count - 1):
                Image.fromarray(pixels).save(
                    args.output_dir / f"source_frame_{number:06d}.jpg",
                    quality=96)
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
    errors = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    if encoder.wait(timeout=120) or len(pts_seen) != count:
        raise RuntimeError(f"Clip failed after {len(pts_seen)} frames: {errors[-1500:]}")
    capture = cv2.VideoCapture(str(target))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded != count or abs(fps - 30) > 0.1 or (width, height) != (1920, 1080):
        raise RuntimeError("Focused clip validation failed")
    record = {
        "schema": "focused-content-clip-v1", "status": "visual_review_required",
        "source": str(args.source), "source_sha256": args.source_sha256,
        "requested_start_seconds": args.start_seconds,
        "source_frame_range_inclusive": [indices[0], indices[-1]],
        "source_pts_seconds_first_last": [pts_seen[0], pts_seen[-1]],
        "output": str(target), "output_sha256": file_hash(target),
        "frames": decoded, "fps": fps, "width": width, "height": height,
        "duration_seconds": decoded / fps,
        "color_conversion": "project fixed HLG to SDR sRGB at -1 EV",
        "role": "focused content train/evaluation clip, not a target-style reference",
    }
    write_json(args.output_dir / "clip.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
