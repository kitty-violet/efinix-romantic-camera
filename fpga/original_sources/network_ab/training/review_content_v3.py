"""Render the complete new validation split and summarize focus statistics."""

import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw

from network_ab.training.data import load_content_manifest
from network_ab.training.runtime import configure_workspace, write_json

ROOT = configure_workspace()
DATA = ROOT / "data/processed/content-20261001-v3-normal-validation/manifest.approved.json"
OUTPUT = ROOT / "runs/style-target-full-timeline-20261001/content-v3-review"


def main():
    manifest = load_content_manifest(DATA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "val", "test"):
        rows = [row for row in manifest["images"] if row["split"] == split]
        scores = sorted(row["laplacian_variance"] for row in rows)
        report[split] = {"count": len(rows), "laplacian_median": statistics.median(scores),
                         "laplacian_p10": scores[len(scores)//10],
                         "sources": sorted({Path(row["source"]).name for row in rows})}
        if split != "val":
            continue
        for start in range(0, len(rows), 30):
            chunk = rows[start:start + 30]
            canvas = Image.new("RGB", (5 * 320, 6 * 204), "#191919")
            pen = ImageDraw.Draw(canvas)
            for slot, row in enumerate(chunk):
                x, y = slot % 5 * 320, slot // 5 * 204
                with Image.open(row["path"]) as im:
                    canvas.paste(im.resize((320, 180), Image.Resampling.BICUBIC), (x, y + 24))
                name = Path(row["source"]).stem
                pen.text((x + 3, y + 3), f"{name} {row['timestamp']:.1f}s lap={row['laplacian_variance']:.0f}",
                         fill="white")
            canvas.save(OUTPUT / f"validation_{start//30:02d}.jpg", quality=92)
    write_json(OUTPUT / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
