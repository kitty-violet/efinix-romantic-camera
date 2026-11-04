"""Read-only dense quality scan of newly uploaded game target recordings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
import imageio_ffmpeg

from .audit_full_video_timeline import sample

ROOT = Path(r"E:\fpga\model-training-workspace")
SOURCE = ROOT / "data/incoming/style-target-replacement"
OUT = ROOT / "runs/style-target-new-sources-20261001"
NAMES = ["Dying Light 2026-10-01 18-54-23.mp4",
         "Dying Light 2026-10-01 18-59-02.mp4"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contacts(rows: list[dict], prefix: str) -> list[str]:
    pages = []
    for first in range(0, len(rows), 16):
        canvas = Image.new("RGB", (1920, 1200), "#181818")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[first:first + 16]):
            x, y = slot % 4 * 480, slot // 4 * 300
            with Image.open(row["thumbnail"]) as thumb:
                canvas.paste(thumb, (x, y + 28))
            draw.text((x + 5, y + 5),
                      f"{row['video'][-15:]} {row['second']:.0f}s L{row['best_3_tile_laplacian']:.0f}",
                      fill="white")
        dest = OUT / f"{prefix}_{first // 16:02d}.jpg"
        canvas.save(dest, quality=94)
        pages.append(str(dest))
    return pages


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = []
    tasks = []
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    (OUT / "thumbnails").mkdir(parents=True)
    for name in NAMES:
        path = SOURCE / name
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        if shape != [1920, 1080] or duration < 30:
            raise ValueError(f"Unexpected source dimensions/duration: {path}")
        source = {"file": str(path), "bytes": path.stat().st_size,
                  "sha256": sha256(path), "duration_seconds": duration,
                  "width": shape[0], "height": shape[1], "fps_estimate": fps}
        sources.append(source)
        times = range(5, int(duration - 2), 5)
        tasks.extend((path, float(second), exe, OUT / "thumbnails") for second in times)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(sample, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 50 == 0:
                print(f"scanned {i}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (row["video"], row["second"]))
    # At least one chronological frame per 30 seconds, independent of sharpness.
    overview = [row for row in rows if row["second"] % 30 == 5]
    # Separately inspect the sharpest frame in each 30-second bin.
    by_bin = {}
    for row in rows:
        key = row["video"], int(row["second"] // 30)
        if key not in by_bin or row["best_3_tile_laplacian"] > by_bin[key]["best_3_tile_laplacian"]:
            by_bin[key] = row
    sharpest = sorted(by_bin.values(), key=lambda row: (row["video"], row["second"]))
    report = {"schema": "new-style-source-survey-v1", "status": "REVIEW_ONLY",
              "sources": sources, "interval_seconds": 5,
              "sample_count": len(rows), "rows": rows,
              "chronological_overview_pages": contacts(overview, "chronological"),
              "sharpest_per_30s_pages": contacts(sharpest, "sharpest"),
              "notes": "Sharpness scores only rank candidates. Review scene, motion blur, UI, weapon, foreground and temporary effects at native resolution."}
    (OUT / "timeline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for source in sources:
        path = Path(source["file"])
        if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
            raise ValueError(f"Source changed while scanning: {path}")
    print(f"frames={len(rows)} chronological_pages={len(report['chronological_overview_pages'])} sharpest_pages={len(report['sharpest_per_30s_pages'])}")


if __name__ == "__main__":
    main()
