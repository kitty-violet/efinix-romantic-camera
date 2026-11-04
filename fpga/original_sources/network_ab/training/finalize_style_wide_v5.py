"""Remove residual overlay-bearing tiles from the wide-first style set."""

import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from network_ab.training.data import audit_style
from network_ab.training.runtime import configure_workspace, file_hash, write_json

ROOT = configure_workspace()
SOURCE = ROOT / "data/processed/style-target-20261001-v4-wide-reviewed"
TARGET = ROOT / "data/processed/style-target-20261001-v5-wide-clean"
REVIEW = ROOT / "runs/style-target-full-timeline-20261001"
EXCLUDED = {
    "T_2_00470_D.png": "visible ammunition counter",
    "T_3_01805_A.png": "visible objective marker",
    "T_3_02250_A.png": "visible objective marker and transient blue glow",
    "T_3_02480_A.png": "visible objective distance marker",
}


def contacts(rows, kind, width, height, columns, page_size):
    items = [row for row in rows if row["kind"] == kind]
    paths = []
    for first in range(0, len(items), page_size):
        chunk = items[first:first + page_size]
        count_rows = (len(chunk) + columns - 1) // columns
        out = Image.new("RGB", (width * columns, (height + 24) * count_rows), "#161616")
        pen = ImageDraw.Draw(out)
        for n, row in enumerate(chunk):
            x, y = n % columns * width, n // columns * (height + 24)
            with Image.open(TARGET / row["file"]) as im:
                out.paste(im.resize((width, height), Image.Resampling.BICUBIC), (x, y + 24))
            pen.text((x + 4, y + 4), row["file"], fill="white")
        path = REVIEW / f"v5_{kind}_contact_{first // page_size:02d}.jpg"
        out.save(path, quality=94)
        paths.append(str(path))
    return paths


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    parent = json.loads((SOURCE / "dataset-manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in parent["records"] if r["file"] not in EXCLUDED]
    if len(rows) != len(parent["records"]) - len(EXCLUDED):
        raise ValueError("One or more exclusions missing")
    TARGET.mkdir(parents=True)
    for row in rows:
        src = SOURCE / row["file"]
        if file_hash(src) != row["sha256"]:
            raise ValueError(f"Hash mismatch: {src}")
        shutil.copy2(src, TARGET / row["file"])
    manifest = dict(parent)
    manifest.update(status="approved", image_count=len(rows),
                    wide_scene_count=sum(r["kind"] == "wide_scene" for r in rows),
                    texture_region_count=sum(r["kind"] == "texture_region" for r in rows),
                    source_times=len({(r["source_video"], r["requested_second"]) for r in rows}),
                    parent_draft=str(SOURCE / "dataset-manifest.json"),
                    final_exclusions=EXCLUDED,
                    records=rows,
                    contact_pages=contacts(rows, "wide_scene", 440, 247, 3, 12)
                    + contacts(rows, "texture_region", 220, 220, 5, 30))
    write_json(TARGET / "dataset-manifest.json", manifest)
    audit_style(TARGET)
    print(json.dumps({k: manifest[k] for k in
                      ("image_count", "wide_scene_count", "texture_region_count", "source_times")}))


if __name__ == "__main__":
    main()
