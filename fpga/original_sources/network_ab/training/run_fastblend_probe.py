"""Evaluate official FastBlend code on the isolated park video frames."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .postprocess_teacher_video import warp_errors
from .runtime import configure_workspace, file_hash, write_json


class Frames:
    def __init__(self, images: list[np.ndarray]):
        self.images = images

    def raw_data(self) -> list[np.ndarray]:
        return self.images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.images[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--mode", choices=("Fast", "Accurate"), default="Fast")
    args = parser.parse_args()
    root = configure_workspace()
    os.environ.setdefault("CUPY_CACHE_DIR", str(root / "cache/cupy"))
    vendor = root / "vendor/sd-webui-fastblend"
    revision = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(vendor))
    import cupy as cp
    from FastBlend.runners import AccurateModeRunner, FastModeRunner

    state = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "rendered_visual_review_required":
        raise ValueError("Source rendering is incomplete")
    if file_hash(Path(state["video"])) != state["video_sha256"]:
        raise ValueError("Source video changed")
    indices = state["source_frame_indices"][:args.count]
    if len(indices) < 2 or args.width % 2:
        raise ValueError("Need at least two frames and an even width")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    guide = []
    style = []
    for index in indices:
        for destination, prefix in ((guide, "source"), (style, "styled")):
            image = Image.open(args.input_dir / f"{prefix}_{index:06d}.png").convert("RGB")
            height = round(image.height * args.width / image.width)
            destination.append(np.asarray(image.resize(
                (args.width, height), Image.Resampling.LANCZOS)))
    cp.random.seed(20261002)
    start = time.perf_counter()
    runner = FastModeRunner() if args.mode == "Fast" else AccurateModeRunner()
    runner.run(
        Frames(guide), Frames(style), batch_size=1,
        window_size=args.window_size,
        ebsynth_config={"minimum_patch_size": 5, "threads_per_block": 8,
                        "num_iter": 3, "gpu_id": 0, "guide_weight": 10.0,
                        "initialize": "identity", "tracking_window_size": 0},
        save_path=str(frames_dir))
    elapsed = time.perf_counter() - start
    corrected = [np.asarray(Image.open(frames_dir / f"{i:05d}.png").convert("RGB"))
                 for i in range(len(indices))]
    height, width = guide[0].shape[:2]
    video = args.output_dir / "source_raw_fastblend.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             state["fps"], (width * 3, height + 24))
    if not writer.isOpened():
        raise RuntimeError("Could not create FastBlend comparison video")
    try:
        for i, (source, raw, smooth) in enumerate(zip(guide, style, corrected)):
            panel = np.zeros((height + 24, width * 3, 3), dtype=np.uint8)
            panel[:] = 21
            panel[24:, :width] = source
            panel[24:, width:2 * width] = raw
            panel[24:, 2 * width:] = smooth
            for x, label in ((12, "SOURCE"), (width + 12, "FRAMEWISE RAW"),
                             (2 * width + 12, "FASTBLEND")):
                cv2.putText(panel, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if i in (0, len(indices) // 2, len(indices) - 1):
                Image.fromarray(panel).save(args.output_dir / f"review_{i:02d}.jpg",
                                            quality=94)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    encoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if encoded != len(indices):
        raise RuntimeError(f"Encoded {encoded} frames, expected {len(indices)}")
    errors = [warp_errors(guide[i - 1], guide[i], style[i - 1], style[i],
                          corrected[i - 1], corrected[i])
              for i in range(1, len(indices))]
    record = {
        "schema": "fastblend-probe-v1", "status": "visual_review_required",
        "vendor": "https://github.com/Artiprocher/sd-webui-fastblend",
        "vendor_revision": revision,
        "source_render": state["video"],
        "source_render_sha256": state["video_sha256"],
        "lora_sha256": state["lora_sha256"],
        "frame_indices": indices, "fps": state["fps"],
        "width": width, "height": height, "window_size": args.window_size,
        "mode": args.mode,
        "batch_size": 1, "minimum_patch_size": 5,
        "num_iter": 3, "guide_weight": 10.0,
        "elapsed_seconds": elapsed,
        "flow_aligned_l1": {key: float(np.mean([row[key] for row in errors]))
                            for key in ("source_l1", "raw_l1", "corrected_l1")},
        "video": str(video), "video_sha256": file_hash(video),
        "limitation": "FastBlend is a postprocessor; visual inspection is required for blur, ghosting, and invented identities.",
    }
    write_json(args.output_dir / "review.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
