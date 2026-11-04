"""Read-only video audit with decoded PTS and review sheets; no class labels inferred."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

from .runtime import configure_workspace, file_hash, write_json
from .media import decode_frames, probe_video


def audit_video(path, output, interval, start=0.0, end=math.inf):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    destination = output / path.stem
    destination.mkdir(parents=True, exist_ok=False)
    decode = probe_video(path)
    reader = decode_frames(path)
    metadata = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
                "width": decode["width"], "height": decode["height"],
                "reported_fps": decode["fps"], "decode": decode,
                "automatic_rotation": decode["autorotate"]}
    frames, count, next_sample, previous_pts, nonmonotonic = [], 0, start, -1.0, 0
    last_pts = None
    try:
        for frame_index, pts, rgb in reader:
            count += 1
            if pts < previous_pts:
                nonmonotonic += 1
            previous_pts, last_pts = pts, pts
            if pts > end:
                break
            if pts + 1e-6 < next_sample:
                continue
            next_sample += interval
            im = Image.fromarray(rgb)
            im.thumbnail((960, 960), Image.Resampling.LANCZOS)
            filename = f"frame_{count-1:06d}_{pts:07.3f}s.jpg"
            im.save(destination / filename, quality=93)
            gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
            frames.append({"frame_index": count-1, "pts_seconds": pts,
                           "path": str(destination / filename),
                           "laplacian_variance_at_preview_size": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           "dark_fraction": float((gray <= 3).mean()),
                           "bright_fraction": float((gray >= 252).mean())})
    finally:
        reader.close()
    metadata.update(decoded_frames=count, last_decoded_pts_seconds=last_pts,
                    nonmonotonic_pts=nonmonotonic, samples=frames,
                    decode_scope="full_video" if math.isinf(end) else [start, end],
                    labels="UNREVIEWED", sampling_interval_seconds=interval)
    def sheet(rows, filename):
        width, height = 384, 246
        canvas = Image.new("RGB", (width*4, 38+height*math.ceil(len(rows)/4)), "#161616")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), f"{path.name} | timestamps = decoded video PTS", fill="white")
        for i, item in enumerate(rows):
            x, y = i % 4 * width, 38 + i // 4 * height
            with Image.open(item["path"]) as im:
                preview = ImageOps.contain(im, (width, 216), Image.Resampling.LANCZOS)
                canvas.paste(preview, (x+(width-preview.width)//2, y))
            draw.text((x+8, y+220), f"{item['pts_seconds']:.3f}s  frame {item['frame_index']}", fill="white")
        canvas.save(destination / filename, quality=94)
    for offset in range(0, len(frames), 20):
        sheet(frames[offset:offset+20], f"contact_{offset//20:02d}.jpg")
    indices = sorted(set(round(i*(len(frames)-1)/15) for i in range(16))) if frames else []
    if indices:
        sheet([frames[i] for i in indices], "overview.jpg")
    write_json(destination / "audit.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("path", "decoded_frames", "last_decoded_pts_seconds", "nonmonotonic_pts")}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=math.inf)
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "data/incoming").resolve() not in args.input.resolve().parents:
        raise ValueError("Input must reside in the authorized upload directory")
    if root not in args.output.resolve().parents or args.interval <= 0 or args.start < 0 or args.end < args.start:
        raise ValueError("Invalid output or sample interval")
    paths = sorted(args.input.rglob("*.mp4")) if args.input.is_dir() else [args.input]
    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda p: audit_video(p, args.output, args.interval, args.start, args.end), paths))
    write_json(args.output / "inventory.json", {"schema": "model-training-upload-review-v1", "status": "visual_review_pending",
               "sources": results, "total_bytes": sum(r["bytes"] for r in results),
               "total_decoded_frames": sum(r["decoded_frames"] for r in results),
               "total_duration_estimate_seconds": sum((r["last_decoded_pts_seconds"] or 0) + (1/r["reported_fps"] if r["reported_fps"] else 0) for r in results)})


if __name__ == "__main__":
    main()
