"""Apply a motion-tracked low-frequency game look to native 1080p video.

The source supplies all geometry and fine detail. Diffusion keyframes supply a
slowly varying Lab style field; no frame is duplicated to extend duration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from .media import decode_frames, decoder_identity, probe_video
from .runtime import file_hash, write_json


def rgb_to_lab(pixels: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).astype(np.float32)


def style_delta(style: np.ndarray, source: np.ndarray, sigma: float) -> np.ndarray:
    if style.shape != source.shape:
        raise ValueError("Style and keyframe source sizes differ")
    delta = rgb_to_lab(style) - rgb_to_lab(source)
    delta = cv2.GaussianBlur(delta, (0, 0), sigma)
    return np.clip(delta, [-48, -25, -25], [48, 25, 25]).astype(np.float32)


def track_field(previous: np.ndarray, current: np.ndarray,
                previous_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp a smooth style field using source motion and reject poor matches."""
    backward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(current, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 21, 3, 5, 1.2, 0)
    height, width = current.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    mx, my = xx + backward[:, :, 0], yy + backward[:, :, 1]
    warp = lambda image: cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
    warped_field = warp(previous_field)
    source_error = np.abs(current.astype(np.float32) -
                          warp(previous).astype(np.float32)).mean(axis=2)
    confidence = np.clip((32.0 - source_error) / 20.0, 0.0, 1.0)
    confidence *= ((mx >= 0) & (mx < width - 1) &
                   (my >= 0) & (my < height - 1)).astype(np.float32)
    confidence = cv2.GaussianBlur(confidence, (0, 0), 2.0)
    return warped_field, confidence


def encode_pipe(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    exe = decoder_identity()["path"]
    return subprocess.Popen([
        exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "16", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart",
        str(path),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--keyframe-source-dir", type=Path, required=True)
    parser.add_argument("--keyframe-style-dir", type=Path, required=True)
    parser.add_argument("--style-prefix", default="final")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--keyframe-step", type=int, default=3)
    parser.add_argument("--keyframe-indices", type=int, nargs="+",
                        help="Explicit native frame indices; overrides keyframe-step")
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--update-beta", type=float, default=0.16)
    parser.add_argument("--style-gain", type=float, default=1.0)
    args = parser.parse_args()
    if (args.end_frame < 2 or args.fps <= 0 or args.keyframe_step <= 0 or
            args.sigma <= 0 or not 0 <= args.update_beta <= 1 or
            not 0 <= args.style_gain <= 2):
        raise ValueError("Invalid video or style parameters")
    source_state = json.loads(args.source_state.read_text(encoding="utf-8"))
    video = Path(source_state["source_video"])
    if file_hash(video) != source_state["source_sha256"]:
        raise ValueError("Holdout video changed")
    probe = probe_video(video)
    if (probe["width"], probe["height"]) != (1920, 1080):
        raise ValueError("Expected native 1920x1080 source")
    indices = (args.keyframe_indices if args.keyframe_indices is not None else
               list(range(0, args.end_frame + 1, args.keyframe_step)))
    if (not indices or indices != sorted(set(indices)) or indices[0] != 0 or
            indices[-1] > args.end_frame):
        raise ValueError("Keyframe indices must be distinct, ascending, start at 0, and fit the clip")
    keyframes = set(indices)
    if not (args.keyframe_source_dir / "source_000000.png").is_file():
        raise FileNotFoundError("Missing first keyframe source")
    for index in indices:
        if not (args.keyframe_source_dir / f"source_{index:06d}.png").is_file():
            raise FileNotFoundError(f"Missing keyframe source {index}")
        if not (args.keyframe_style_dir /
                f"{args.style_prefix}_{index:06d}.png").is_file():
            raise FileNotFoundError(f"Missing styled keyframe {index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "native_detail_stable_1080p.mp4"
    encoder = encode_pipe(target, 1920, 1080, args.fps)
    previous_small = field = None
    changes = []
    decoded = 0
    try:
        for index, pts, native in decode_frames(
                video, indices=list(range(args.end_frame + 1)),
                hlg_exposure_stops=-1.0):
            small = cv2.resize(native, (768, 432), interpolation=cv2.INTER_AREA)
            candidate = None
            if index in keyframes:
                key_source = np.asarray(Image.open(
                    args.keyframe_source_dir / f"source_{index:06d}.png").convert("RGB"))
                key_style = np.asarray(Image.open(
                    args.keyframe_style_dir /
                    f"{args.style_prefix}_{index:06d}.png").convert("RGB"))
                candidate = style_delta(key_style, key_source, args.sigma)
            if field is None:
                field = candidate
            else:
                warped, confidence = track_field(previous_small, small, field)
                if candidate is not None:
                    keep = (1.0 - args.update_beta) * confidence[:, :, None]
                    field = warped * keep + candidate * (1.0 - keep)
                else:
                    mean = field.reshape(-1, 3).mean(axis=0)
                    field = warped * confidence[:, :, None] + mean * (1.0 - confidence[:, :, None])
            # Fine structure comes from the native source, never a 768px upscale.
            native_lab = rgb_to_lab(native)
            full_delta = cv2.resize(field, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            output_lab = np.clip(native_lab + args.style_gain * full_delta,
                                 0, 255).astype(np.uint8)
            output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
            if index in (0, args.end_frame // 2, args.end_frame):
                Image.fromarray(output).save(
                    args.output_dir / f"review_{index:06d}.jpg", quality=95)
            if encoder.stdin is None:
                raise RuntimeError("Encoder stdin missing")
            encoder.stdin.write(output.tobytes())
            decoded += 1
            changes.append(float(np.abs(output.astype(np.float32) -
                                        native.astype(np.float32)).mean()))
            if index % 30 == 0 or index == args.end_frame:
                print(f"frame {index + 1}/{args.end_frame + 1} {pts:.3f}s", flush=True)
            previous_small = small
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
    errors = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    if encoder.wait(timeout=120) or decoded != args.end_frame + 1:
        raise RuntimeError(f"1080p encoder failed after {decoded} frames: {errors[-2000:]}")
    capture = cv2.VideoCapture(str(target))
    encoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    playback_fps = float(capture.get(cv2.CAP_PROP_FPS))
    verified = sum(capture.read()[0] for _ in range(encoded))
    capture.release()
    if encoded != decoded or verified != decoded or abs(playback_fps - args.fps) > 0.1:
        raise RuntimeError("Encoded video did not decode fully at the requested fps")
    record = {
        "schema": "native-detail-stable-video-v1", "status": "visual_review_required",
        "source_video": str(video), "source_sha256": source_state["source_sha256"],
        "source_resolution": [1920, 1080], "source_fps": probe["fps"],
        "frames": encoded, "fps": playback_fps, "duration_seconds": encoded / playback_fps,
        "keyframe_step": args.keyframe_step if args.keyframe_indices is None else None,
        "keyframe_indices": indices, "keyframes": len(indices),
        "keyframe_style_dir": str(args.keyframe_style_dir),
        "style_prefix": args.style_prefix, "sigma": args.sigma,
        "update_beta": args.update_beta, "style_gain": args.style_gain,
        "mean_rgb_change_from_source": float(np.mean(changes)),
        "video": str(target), "video_sha256": file_hash(target),
        "limitation": "Native 1080p detail and identity are preserved; diffusion contributes low-frequency look only, not native-resolution game geometry or textures.",
    }
    write_json(args.output_dir / "review.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
