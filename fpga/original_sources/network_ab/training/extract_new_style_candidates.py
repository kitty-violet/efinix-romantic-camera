"""Decode native 1280x720 candidates from visually identified gameplay windows.

This produces review material only. A separate explicit approval pass is required
before frames can be copied into a training target dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from .review_new_style_sources import sha256

ROOT = Path(r"E:\fpga\model-training-workspace")
SCAN = ROOT / "runs/style-target-new-sources-20261001/timeline.json"
OUT = ROOT / "runs/style-target-new-sources-20261001/native-candidates"
CROP = (0, 125, 1280, 720)

# Identified by inspecting the complete chronological contact sheets and all
# 275 cropped gameplay previews. Neighbouring frames are intentional: a later
# human visual pass picks the least obstructed/sharpest one.
TIMES = {
    "18-54-23": [75, 80, 85, 90, 110, 115, 120, 125, 150, 155, 160],
    "18-59-02": [10, 20, 35, 45, 55, 65, 105, 110, 120, 125,
                  130, 175, 180, 205, 215, 235, 240, 260, 270,
                  280, 605, 615, 625, 645, 650, 660, 665, 675,
                  680, 690, 695, 700, 750, 755, 765, 770, 780,
                  840, 845, 850, 855, 1045, 1050, 1070, 1090,
                  1125, 1130, 1150, 1165, 1220, 1455, 1465,
                  1485, 1500, 1615, 1685, 1705, 1985, 2065,
                  2070, 2075, 2080],
}


def decode(video: Path, second: int, dest: Path, exe: str) -> dict:
    x, y, w, h = CROP
    command = [exe, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", str(second), "-i", str(video), "-map", "0:v:0",
               "-frames:v", "1", "-vf", f"crop={w}:{h}:{x}:{y},format=rgb24",
               "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=90)
    if result.returncode or not result.stdout:
        raise RuntimeError(f"Decode failed: {video.name} {second}: {result.stderr[-200:]!r}")
    dest.write_bytes(result.stdout)
    with Image.open(dest) as image:
        if image.size != (w, h):
            raise ValueError(f"Unexpected output size: {dest}")
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # Focus on scene texture. Sky and near foreground alone can mislead.
    tiles = []
    for yy in range(0, h, 180):
        for xx in range(0, w, 320):
            tiles.append(float(lap[yy:yy + 180, xx:xx + 320].var()))
    return {"file": dest.name, "sha256": sha256(dest),
            "laplacian_variance_native": float(lap.var()),
            "best_4_tile_laplacian_native": float(np.mean(sorted(tiles, reverse=True)[:4])),
            "luma_mean": float(gray.mean()),
            "dark_fraction": float((gray < 10).mean()),
            "bright_fraction": float((gray > 245).mean())}


def contacts(rows: list[dict]) -> list[str]:
    pages = []
    for first in range(0, len(rows), 16):
        canvas = Image.new("RGB", (1680, 1072), "#171717")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[first:first + 16]):
            x, y = slot % 4 * 420, slot // 4 * 268
            with Image.open(OUT / row["file"]) as src:
                canvas.paste(src.resize((420, 236), Image.Resampling.LANCZOS), (x, y + 28))
            label = f"{row['source_tag']} {row['second']}s L{row['laplacian_variance_native']:.0f}"
            draw.text((x + 5, y + 5), label, fill="white")
        dest = OUT / f"candidate_{first // 16:02d}.jpg"
        canvas.save(dest, quality=94)
        pages.append(str(dest))
    return pages


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    sources = {}
    for row in scan["sources"]:
        tag = "18-54-23" if "18-54-23" in row["file"] else "18-59-02"
        path = Path(row["file"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Source changed: {path}")
        sources[tag] = row
    OUT.mkdir(parents=True)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    rows = []
    total = sum(map(len, TIMES.values()))
    for tag, times in TIMES.items():
        video = Path(sources[tag]["file"])
        for second in times:
            dest = OUT / f"N_{tag}_{second:05d}.png"
            row = {"source_tag": tag, "source_file": str(video),
                   "source_sha256": sources[tag]["sha256"],
                   "second": second, "crop_xywh_native": list(CROP),
                   "decision": "pending_visual_review"}
            row.update(decode(video, second, dest, exe))
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"decoded {len(rows)}/{total}", flush=True)
    report = {"schema": "new-style-native-candidates-v1",
              "status": "DRAFT_NOT_FOR_TRAINING", "source_scan": str(SCAN),
              "crop_xywh_native": list(CROP), "rows": rows,
              "contact_pages": contacts(rows)}
    (OUT / "candidate-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidates={len(rows)} contacts={len(report['contact_pages'])}")


if __name__ == "__main__":
    main()
