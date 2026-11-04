"""Review the short, newly uploaded third target video at native crop size."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw

from .extract_new_style_candidates import decode
from .review_new_style_sources import sha256

ROOT = Path(r"E:\fpga\model-training-workspace")
SOURCE = ROOT / "data/incoming/style-target-replacement/屏幕录制 2026-10-01 194608.mp4"
OUT = ROOT / "runs/style-target-new-sources-20261001/latest-video-review"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    size = SOURCE.stat().st_size
    source_hash = sha256(SOURCE)
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise ValueError(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width < 1280 or height < 845 or duration < 10:
        raise ValueError(f"Unexpected source shape/duration: {width}x{height}, {duration}s")
    OUT.mkdir(parents=True)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    rows = []
    for second in range(1, int(duration - 1), 2):
        dest = OUT / f"N_latest_{second:03d}.png"
        row = {"video": SOURCE.name, "source_sha256": source_hash,
               "second": second, "crop_xywh_native": [0, 125, 1280, 720],
               "decision": "pending_visual_review"}
        row.update(decode(SOURCE, second, dest, exe))
        rows.append(row)
    pages = []
    for first in range(0, len(rows), 16):
        canvas = Image.new("RGB", (1680, 1072), "#171717")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[first:first + 16]):
            x, y = slot % 4 * 420, slot // 4 * 268
            with Image.open(OUT / row["file"]) as src:
                canvas.paste(src.resize((420, 236), Image.Resampling.LANCZOS), (x, y + 28))
            draw.text((x + 5, y + 5), f"latest {row['second']}s L{row['laplacian_variance_native']:.0f}", fill="white")
        dest = OUT / f"candidate_{first // 16:02d}.jpg"
        canvas.save(dest, quality=94)
        pages.append(str(dest))
    if SOURCE.stat().st_size != size or sha256(SOURCE) != source_hash:
        raise ValueError("Source changed during review")
    report = {"schema": "latest-style-source-review-v1", "status": "DRAFT_NOT_FOR_TRAINING",
              "source": {"file": str(SOURCE), "sha256": source_hash, "bytes": size,
                         "width": width, "height": height, "fps": fps,
                         "duration_seconds": duration},
              "sampling_seconds": 2, "rows": rows, "contact_pages": pages}
    (OUT / "candidate-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"latest: {duration:.3f}s, {len(rows)} frames, {len(pages)} contacts")


if __name__ == "__main__":
    main()
