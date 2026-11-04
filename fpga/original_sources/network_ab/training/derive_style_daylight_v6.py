"""Derive a controlled daylight target subset from the already reviewed v5 set."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from .data import audit_style
from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace() / "data/processed"
    parent = root / "style-target-20261001-v5-wide-clean"
    target = root / "style-target-20261001-v6-daylight-draft"
    audit_style(parent)
    old_path = parent / "dataset-manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if target.exists():
        raise FileExistsError(target)
    # This isolates the warm dusk sequence beginning around 32 minutes of 3.mp4.
    records = [copy.deepcopy(row) for row in old["records"]
               if row["source_video"] in ("1.mp4", "2.mp4")
               or (row["source_video"] == "3.mp4" and row["requested_second"] <= 1880)]
    if sum(r["kind"] == "wide_scene" for r in records) != 6:
        raise ValueError("Expected six wide daylight/interior scenes")
    target.mkdir(parents=True)
    for row in records:
        shutil.copy2(parent / row["file"], target / row["file"])
        if file_hash(target / row["file"]) != row["sha256"]:
            raise ValueError("Copied target hash changed")
    contacts = []
    review_dir = target / "review"
    review_dir.mkdir()
    for kind in ("wide_scene", "texture_region"):
        selected = [row for row in records if row["kind"] == kind]
        canvas = Image.new("RGB", (4 * 320, ((len(selected) + 3) // 4) * 205), "#101010")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(selected):
            x, y = i % 4 * 320, i // 4 * 205
            with Image.open(target / row["file"]) as image:
                canvas.paste(image.convert("RGB").resize((320, 180)), (x, y + 20))
            draw.text((x + 4, y + 2), row["file"], fill="white")
        path = review_dir / f"contact_{kind}.jpg"
        canvas.save(path, quality=94)
        contacts.append(str(path))
    manifest = copy.deepcopy(old)
    manifest.update(status="draft_requires_visual_review", image_count=len(records),
        wide_scene_count=sum(r["kind"] == "wide_scene" for r in records),
        texture_region_count=sum(r["kind"] == "texture_region" for r in records),
        source_times=sorted({f"{r['source_video']}:{r['requested_second']}" for r in records}),
        records=records, parent_manifest=str(old_path), parent_manifest_sha256=file_hash(old_path),
        contact_pages=contacts,
        selection_policy="Same native reviewed images; omit 3.mp4 frames after 1880s to isolate daylight/interior target palette",
        selection_purpose="Controlled CUT ablation for dusk-driven hue drift; not a final style approval")
    write_json(target / "dataset-manifest.draft.json", manifest)
    print(json.dumps({"target": str(target), "images": len(records),
                      "wide": 6, "texture": len(records)-6, "contacts": contacts}, indent=2))


if __name__ == "__main__":
    main()
