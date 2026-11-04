"""Inspect native neighboring game frames around six reviewed wide scenes.

Extraction is read-only with respect to uploaded videos and approved targets.
No candidate becomes training data without a separate visual selection.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from .runtime import configure_workspace, file_hash, write_json


CROP = (180, 125, 1320, 742)
OFFSETS = (-20, -15, -10, -5, 5, 10, 15, 20)


def extract(item):
    source, center, second, expected_sha, output = item
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-nostdin", "-ss", str(second), "-i", str(source), "-map", "0:v:0",
               "-frames:v", "1", "-vf", "format=rgb24", "-f", "image2pipe",
               "-vcodec", "png", "pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=90)
    if result.returncode or not result.stdout:
        raise RuntimeError(f"Frame decode failed: {source.name} {second}")
    with Image.open(BytesIO(result.stdout)) as loaded:
        frame = loaded.convert("RGB")
    if frame.size != (1918, 1078):
        raise ValueError("Unexpected native game capture dimensions")
    x, y, w, h = CROP
    image = frame.crop((x, y, x + w, y + h))
    path = output / f"{source.stem}_{second:05d}.png"
    image.save(path)
    pixels = np.asarray(image)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    return {"source": str(source), "source_sha256": expected_sha,
            "center_second": center, "requested_second": second,
            "path": str(path), "sha256": file_hash(path), "crop_xywh_native": CROP,
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "bright_fraction": float((gray >= 245).mean())}


def main():
    root = configure_workspace()
    parent_path = root / "data/processed/style-target-20261001-v6-daylight-draft/dataset-manifest.json"
    manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    centers = [(row["source_video"], row["requested_second"])
               for row in manifest["records"] if row["kind"] == "wide_scene"]
    if len(centers) != 6:
        raise ValueError("Expected six curated wide-scene centers")
    folder = root / "runs/style-neighbor-audit-20261001"
    if folder.exists():
        raise FileExistsError(folder)
    candidate_dir = folder / "candidates"
    candidate_dir.mkdir(parents=True)
    source_root = root / "data/incoming/style-target-replacement"
    source_hashes = {row["source_video"]: row["source_sha256"]
                     for row in manifest["records"]}
    tasks = []
    for name, center in centers:
        source = source_root / name
        if file_hash(source) != source_hashes[name]:
            raise ValueError("Uploaded game video changed")
        tasks.extend((source, center, center + offset, source_hashes[name], candidate_dir)
                     for offset in OFFSETS)
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(extract, item) for item in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda r: (Path(r["source"]).name, r["center_second"],
                             r["requested_second"]))
    contacts = []
    for name, center in centers:
        selected = [row for row in rows if Path(row["source"]).name == name
                    and row["center_second"] == center]
        page = Image.new("RGB", (4 * 330, 2 * 215), "#171717")
        draw = ImageDraw.Draw(page)
        for i, row in enumerate(selected):
            x, y = i % 4 * 330, i // 4 * 215
            with Image.open(row["path"]) as loaded:
                page.paste(loaded.resize((330, 186)), (x, y + 24))
            draw.text((x + 5, y + 4), f"{name} {row['requested_second']}s L{row['laplacian_variance']:.0f}",
                      fill="white")
        path = folder / f"contact_{Path(name).stem}_{center:05d}.jpg"
        page.save(path, quality=94)
        contacts.append(str(path))
    write_json(folder / "audit.json", {"status": "REVIEW_ONLY",
        "parent_manifest": str(parent_path), "parent_manifest_sha256": file_hash(parent_path),
        "candidates": rows, "contact_pages": contacts,
        "policy": "native 1320x742 game pixels; no automatic training approval"})
    print(json.dumps({"candidates": len(rows), "contacts": contacts}, indent=2))


if __name__ == "__main__":
    main()
