"""Make an auditable contact sheet of approved daylight game scenes.

The sheet supports manual selection for the first, vehicle-light style trial.
It does not create or approve a dataset by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    root = configure_workspace()
    manifest = root / "data/processed/style-target-20261001-v14-expanded-reviewed/dataset-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["status"] != "approved" or len(data["records"]) != data["image_count"]:
        raise ValueError("Invalid approved style manifest")
    rows = [row for row in data["records"]
            if row.get("lighting") == "daylight" or row.get("scene", "").startswith("daylight")]
    if len(rows) != 56:
        raise ValueError("Daylight target count changed")
    out = root / "runs/simple-scene-stage-20261002"
    out.mkdir(parents=True, exist_ok=True)
    for page_start in range(0, len(rows), 28):
        page = rows[page_start:page_start + 28]
        sheet = Image.new("RGB", (4 * 320, 7 * 205), "#161616")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(page):
            path = manifest.parent / row["file"]
            if not path.is_file() or file_hash(path) != row["sha256"]:
                raise ValueError(f"Target changed: {path}")
            x, y = i % 4 * 320, i // 4 * 205
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize((320, 180)), (x, y + 22))
            draw.text((x + 5, y + 4), f'{page_start + i:02d} {row["file"]}', fill="white")
        sheet.save(out / f"daylight_candidates_{page_start // 28:02d}.jpg", quality=93)
    write_json(out / "candidate_index.json", {
        "schema": "simple-stage-candidate-index-v1",
        "status": "REVIEW_ONLY",
        "manifest": str(manifest), "manifest_sha256": file_hash(manifest),
        "candidate_count": len(rows),
        "rows": [{"file": row["file"], "sha256": row["sha256"],
                  "scene": row.get("scene"), "source_video": row["source_video"],
                  "second": row["second"]} for row in rows],
    })
    print(out, flush=True)


if __name__ == "__main__":
    main()
