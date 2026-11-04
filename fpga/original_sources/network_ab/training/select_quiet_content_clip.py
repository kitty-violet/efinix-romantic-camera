"""Rank 7-second source clips by camera motion and curate review sheets.

This only proposes candidates. People count and suitability still need visual review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .runtime import write_json


def small_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)


def camera_motion(before: np.ndarray, after: np.ndarray) -> float:
    corners = cv2.goodFeaturesToTrack(before, 240, 0.015, 7)
    if corners is None or len(corners) < 15:
        return float("inf")
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(before, after, corners, None)
    good = status.ravel() != 0
    if good.sum() < 15:
        return float("inf")
    matrix, inliers = cv2.estimateAffinePartial2D(
        corners[good].reshape(-1, 2), tracked[good].reshape(-1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if matrix is None or inliers is None or int(inliers.sum()) < 12:
        return float("inf")
    yy, xx = np.mgrid[0:180:45, 0:320:80].astype(np.float32)
    points = np.stack((xx.ravel(), yy.ravel()), axis=1)
    transformed = points @ matrix[:, :2].T + matrix[:, 2]
    return float(np.median(np.linalg.norm(transformed - points, axis=1)))


def contact_sheet(rows: list[dict], path: Path, caption: str) -> None:
    selected = rows[::max(1, (len(rows) - 1) // 5)][:6]
    w, h = 360, 203
    sheet = Image.new("RGB", (w * 3, (h + 28) * 2 + 30), "#1c1c1c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    draw.text((8, 3), caption, fill="white", font=font)
    for slot, row in enumerate(selected):
        x = (slot % 3) * w
        y = (slot // 3) * (h + 28) + 30
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
        sheet.paste(image, (x, y + 28))
        draw.text((x + 8, y + 2), f'{row["timestamp"]:.1f}s',
                  fill="white", font=font)
    sheet.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 10:
        raise ValueError("Clip length must be 5–10 seconds")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in manifest["images"]:
        groups[row["source"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for source, rows in groups.items():
        rows.sort(key=lambda r: r["timestamp"])
        grays = [small_gray(Path(row["path"])) for row in rows]
        motions = [camera_motion(grays[i], grays[i + 1])
                   if rows[i + 1]["timestamp"] - rows[i]["timestamp"] <= 1.2
                   else float("inf") for i in range(len(rows) - 1)]
        for begin in range(len(rows) - args.seconds + 1):
            window = rows[begin:begin + args.seconds]
            window_motion = motions[begin:begin + args.seconds - 1]
            if any(not np.isfinite(value) for value in window_motion):
                continue
            lap = float(np.median([r["laplacian_variance"] for r in window]))
            bright = float(np.median([r["bright_fraction"] for r in window]))
            if lap < 250 or bright > 0.20:
                continue
            motion = float(np.median(window_motion))
            score = motion + 10 * max(0, bright - 0.08) + 0.005 * max(0, lap - 1500)
            candidates.append({
                "source": source, "source_sha256": window[0]["source_sha256"],
                "split": window[0]["split"],
                "start_seconds": window[0]["timestamp"],
                "end_seconds": window[-1]["timestamp"] + 1.0,
                "median_camera_motion_320px": motion,
                "median_laplacian_1280px": lap,
                "median_bright_fraction": bright,
                "score": score, "samples": [r["path"] for r in window],
            })
    candidates.sort(key=lambda row: row["score"])
    for rank, row in enumerate(candidates[:24]):
        matching = [r for r in groups[row["source"]]
                    if row["start_seconds"] <= r["timestamp"] < row["end_seconds"]]
        path = args.output_dir / f"candidate_{rank:02d}.jpg"
        contact_sheet(matching, path,
                      f'{Path(row["source"]).stem} {row["start_seconds"]:.1f}-{row["end_seconds"]:.1f}s motion {row["median_camera_motion_320px"]:.1f}')
        row["contact_sheet"] = str(path)
    write_json(args.output_dir / "candidates.json", {
        "schema": "quiet-content-clip-candidates-v1",
        "status": "automatic_rank_requires_visual_review",
        "input_manifest": str(args.manifest), "seconds": args.seconds,
        "candidates": candidates,
    })
    print(f"Ranked {len(candidates)} clips; wrote {min(24, len(candidates))} review sheets")
    for row in candidates[:24]:
        print(Path(row["source"]).name, f'{row["start_seconds"]:.1f}-{row["end_seconds"]:.1f}',
              row["split"], f'motion={row["median_camera_motion_320px"]:.1f}',
              f'lap={row["median_laplacian_1280px"]:.0f}',
              f'bright={row["median_bright_fraction"]:.3f}')


if __name__ == "__main__":
    main()
