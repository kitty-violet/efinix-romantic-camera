"""Make crop-view contacts for visual screening of the two new recordings.

Reads the source survey thumbnails; no frames are admitted to training here.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(r"E:\fpga\model-training-workspace")
SCAN = ROOT / "runs/style-target-new-sources-20261001/timeline.json"
OUT = ROOT / "runs/style-target-new-sources-20261001/crop-review-v2"

# Gameplay intervals found by chronologically reviewing the complete 5 s survey.
WINDOWS = {
    "18-54-23": [(75, 170)],
    "18-59-02": [(5, 280), (605, 880), (1000, 1240),
                  (1440, 1500), (1610, 1790), (1870, 2090)],
}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    rows = []
    for row in scan["rows"]:
        tag = "18-54-23" if "18-54-23" in row["video"] else "18-59-02"
        if any(lo <= row["second"] <= hi for lo, hi in WINDOWS[tag]):
            rows.append(row)
    pages = []
    for first in range(0, len(rows), 16):
        canvas = Image.new("RGB", (1680, 1072), "#151515")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[first:first + 16]):
            x, y = slot % 4 * 420, slot // 4 * 268
            with Image.open(row["thumbnail"]) as image:
                # Thumbnail is 480x270 (1/4 native resolution).
                # This previews native crop x=0,y=125,w=1280,h=720.
                if image.size != (480, 270):
                    raise ValueError(f"Unexpected thumbnail shape: {image.size}")
                crop = image.crop((0, 31, 320, 211))
                canvas.paste(crop.resize((420, 236), Image.Resampling.LANCZOS), (x, y + 27))
            label = f"{row['video'][-15:]} {row['second']:.0f}s L{row['best_3_tile_laplacian']:.0f}"
            draw.text((x + 5, y + 5), label, fill="white")
        dest = OUT / f"crop_{first // 16:02d}.jpg"
        canvas.save(dest, quality=94)
        pages.append(str(dest))
    (OUT / "index.json").write_text(json.dumps({
        "schema": "new-style-crop-review-v1", "status": "REVIEW_ONLY",
        "crop_preview_native_xywh": [0, 125, 1280, 720],
        "windows": WINDOWS, "sample_count": len(rows),
        "sample_keys": [{"video": r["video"], "second": r["second"]} for r in rows],
        "pages": pages,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"crop review: {len(rows)} samples, {len(pages)} pages")


if __name__ == "__main__":
    main()
