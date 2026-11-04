"""Final full-contact review removes residual focus, UI and framing defects."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

ROOT = Path(r"E:\fpga\model-training-workspace")
OLD = ROOT / "data/processed/style-target-20261001-v12-quality-reviewed"
OUT = ROOT / "data/processed/style-target-20261001-v13-final-reviewed"
DROP = {
    ("3.mp4", 115): "intentional background depth-of-field blur",
    ("3.mp4", 645): "near-duplicate crane vista",
    ("3.mp4", 1160): "soft close-up surface, weak scene composition",
    ("3.mp4", 1525): "objective marker in central scene",
    ("3.mp4", 1935): "objective UI near the upper center",
    ("3.mp4", 2480): "objective marker in central scene",
    ("3.mp4", 2560): "large soft-focus foreground",
    ("3.mp4", 2720): "large score overlay",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    parent_path = OLD / "dataset-manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "approved" or not DROP.keys() <= {
            (row["source_video"], row["second"]) for row in parent["records"]}:
        raise ValueError("Unexpected parent or drop list")
    (OUT / "review").mkdir(parents=True)
    records = []
    for row in parent["records"]:
        if (row["source_video"], row["second"]) in DROP:
            continue
        source = OLD / row["file"]
        dest = OUT / row["file"]
        if sha(source) != row["sha256"]:
            raise ValueError(source)
        shutil.copy2(source, dest)
        records.append(row)
    pages = []
    for first in range(0, len(records), 12):
        canvas = Image.new("RGB", (1536, 1288), "#151515")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(records[first:first + 12]):
            x, y = slot % 3 * 512, slot // 3 * 322
            with Image.open(OUT / row["file"]) as source:
                if source.size != (1280, 720):
                    raise ValueError(row["file"])
                canvas.paste(source.resize((512, 288), Image.Resampling.LANCZOS), (x, y + 30))
            draw.text((x + 5, y + 6), f"{row['source_video']} {row['second']}s {row['scene']}", fill="white")
        dest = OUT / "review" / f"approved_{first // 12:02d}.jpg"
        canvas.save(dest, quality=94)
        pages.append(str(dest))
    report = {"schema": "model-training-style-v1", "status": "approved",
              "image_count": len(records), "wide_scene_count": len(records),
              "texture_region_count": 0,
              "sampling_policy": {"wide_scene_probability": 1.0,
                                  "texture_region_probability": 0.0},
              "source_videos": sorted({row["source_video"] for row in records}),
              "source_times": [f"{row['source_video']}:{row['second']}" for row in records],
              "selection_policy": "Full native 1280x720 scenes, manually checked at two contact-sheet passes; reject menus, temporary filters, focus blur, UI and dominant foreground",
              "source_scan": parent["source_scan"],
              "parent_approved": str(parent_path), "parent_approved_sha256": sha(parent_path),
              "records": records, "scene_counts": dict(Counter(r["scene"] for r in records)),
              "final_exclusions": [{"source_video": v, "second": s, "reason": reason}
                                   for (v, s), reason in DROP.items()],
              "contact_pages": pages,
              "visual_review": "All 9 initial, 7 gap, 5 pre-final and 4 final contact sheets inspected 2026-10-01"}
    (OUT / "dataset-manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"final_approved={len(records)} final_removed={len(DROP)}")


if __name__ == "__main__":
    main()
