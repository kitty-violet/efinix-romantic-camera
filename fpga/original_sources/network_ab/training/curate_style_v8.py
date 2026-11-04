"""Create native-resolution review candidates from the existing full timeline audit.

The uploaded recordings and earlier approved datasets are never modified. Every
candidate still requires visual approval before it can enter a training set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(r"E:\fpga\model-training-workspace")
SCAN = ROOT / "runs/style-target-full-timeline-20261001/timeline.json"
INVENTORY = ROOT / "runs/style-target-review-20261001/source_inventory.json"
OUT = ROOT / "data/processed/style-target-20261001-v8-quality-draft"
CROP = (180, 125, 1320, 742)

# Windows identified by reviewing the complete 30-second panorama contact sheets.
# The underlying 5-second scan selects the sharpest frame in each 20-second bin.
WINDOWS = {
    "2.mp4": [
        (5, 40, "indoor"), (125, 150, "indoor"),
        (210, 270, "indoor"), (420, 455, "indoor"),
        (565, 605, "indoor"), (900, 950, "indoor"),
        (965, 990, "indoor"),
    ],
    "3.mp4": [
        (75, 120, "indoor"), (175, 235, "daylight-rooftop"),
        (305, 330, "daylight-rooftop"), (370, 420, "indoor"),
        (515, 720, "daylight-rooftop"), (760, 780, "indoor"),
        (905, 925, "indoor"), (1020, 1045, "indoor"),
        (1085, 1105, "daylight-street"), (1140, 1250, "daylight-street"),
        (1470, 1530, "daylight-street"), (1570, 1680, "daylight-street"),
        (1740, 1810, "daylight-rooftop"), (1870, 1950, "daylight-street"),
        (2010, 2065, "daylight-rooftop"), (2280, 2320, "daylight-street"),
        (2370, 2450, "daylight-street"), (2480, 2515, "interior"),
        (2550, 2610, "dusk-rooftop"), (2790, 2860, "dusk-rail"),
        (2880, 2940, "dusk-rail"), (2970, 3055, "dusk-rail"),
        (3125, 3215, "dusk-rail"), (3270, 3300, "dusk-rooftop"),
        (3360, 3415, "dusk-rooftop"), (3450, 3535, "dusk-street"),
        (3570, 3620, "dusk-street"), (3635, 3660, "indoor"),
        (3690, 3730, "indoor"), (3750, 3830, "night-street"),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose(rows: list[dict]) -> list[dict]:
    chosen = []
    for video, windows in WINDOWS.items():
        for lo, hi, scene in windows:
            for start in range(lo, hi + 1, 20):
                eligible = [r for r in rows if r["video"] == video
                            and start <= r["second"] < min(start + 20, hi + 1)]
                if eligible:
                    best = max(eligible, key=lambda r: r["best_3_tile_laplacian"])
                    chosen.append((best, scene))
    unique = {}
    for row, scene in chosen:
        unique.setdefault((row["video"], row["second"]), (row, scene))
    return [{"video": video, "second": second, "scene": unique[(video, second)][1],
             "scan_best3_laplacian": unique[(video, second)][0]["best_3_tile_laplacian"]}
            for video, second in sorted(unique)]


def decode(video: Path, second: float, dest: Path, exe: str) -> dict:
    x, y, w, h = CROP
    command = [exe, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{second:.3f}", "-i", str(video), "-map", "0:v:0",
               "-frames:v", "1", "-vf", f"crop={w}:{h}:{x}:{y},format=rgb24",
               "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=90)
    if result.returncode or not result.stdout:
        raise RuntimeError(f"Decode failed: {video.name} {second}: {result.stderr[-200:]!r}")
    dest.write_bytes(result.stdout)
    with Image.open(dest) as loaded:
        if loaded.size != (w, h):
            raise ValueError(f"Wrong image size: {dest}")
        rgb = np.asarray(loaded.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return {"laplacian_variance_native": float(lap.var()),
            "luma_mean": float(gray.mean()),
            "dark_fraction": float((gray < 10).mean()),
            "bright_fraction": float((gray > 245).mean()),
            "sha256": sha256(dest)}


def contacts(rows: list[dict]) -> list[str]:
    pages = []
    for first in range(0, len(rows), 16):
        canvas = Image.new("RGB", (4 * 420, 4 * 268), "#151515")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[first:first + 16]):
            x, y = slot % 4 * 420, slot // 4 * 268
            with Image.open(OUT / row["file"]) as src:
                canvas.paste(src.resize((420, 236), Image.Resampling.LANCZOS), (x, y + 28))
            label = f"{row['video']} {row['second']:.0f}s {row['scene']} L{row['laplacian_variance_native']:.0f}"
            draw.text((x + 4, y + 5), label, fill="white")
        dest = OUT / "review" / f"candidate_{first // 16:02d}.jpg"
        canvas.save(dest, quality=93)
        pages.append(str(dest))
    return pages


def main() -> None:
    global OUT, CROP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-crop", action="store_true",
                        help="Use the native 1280x720 left crop in a separate v9 draft")
    args = parser.parse_args()
    if args.left_crop:
        OUT = ROOT / "data/processed/style-target-20261001-v9-leftcrop-draft"
        CROP = (0, 125, 1280, 720)
    if OUT.exists():
        raise FileExistsError(OUT)
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    sources = {Path(r["file"]).name: r for r in inventory["sources"]}
    for name in WINDOWS:
        source = Path(sources[name]["file"])
        if sha256(source) != sources[name]["sha256"]:
            raise ValueError(f"Source changed: {source}")
    (OUT / "review").mkdir(parents=True)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    rows = choose(scan["rows"])
    for index, row in enumerate(rows, 1):
        path = OUT / f"C_{Path(row['video']).stem}_{int(row['second']):05d}.png"
        row["file"] = path.name
        row["crop_xywh_native"] = list(CROP)
        row["source_sha256"] = sources[row["video"]]["sha256"]
        row.update(decode(Path(sources[row["video"]]["file"]), row["second"], path, exe))
        row["decision"] = "pending_visual_review"
        if index % 16 == 0:
            print(f"decoded {index}/{len(rows)}", flush=True)
    report = {"schema": "style-target-quality-candidates-v8",
              "status": "DRAFT_NOT_FOR_TRAINING", "source_scan": str(SCAN),
              "source_inventory": str(INVENTORY), "crop_xywh_native": list(CROP),
              "selection": "manual gameplay windows; sharpest scanned 5-second frame per 20-second bin; all require native visual review",
              "excluded_video_1": "Aerial/cinematic scenes, dark loading screens, Windows desktop; no matching clean first-person material",
              "rows": rows, "contact_pages": contacts(rows)}
    (OUT / "review" / "candidate-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidates={len(rows)} contacts={len(report['contact_pages'])}", flush=True)


if __name__ == "__main__":
    main()
