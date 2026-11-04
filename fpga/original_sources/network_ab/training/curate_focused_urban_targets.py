"""Reviewable daytime urban-material subset for the focused Dying Light probe."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .runtime import configure_workspace, file_hash, write_json


# Deliberately exclude panoramas, vehicles, dense foliage, and sunset/night shots.
FILES = (
    "C_3_01245.png", "G_3_01375.png", "C_3_01880.png",
    "G_3_02260.png", "G_3_02275.png", "N_18-54-23_00125.png",
    "N_18-59-02_00055.png", "N_18-59-02_00175.png",
    "N_18-59-02_00205.png", "N_18-59-02_00605.png",
    "N_18-59-02_00625.png",
)
# HUD elements occupy the frame edges; the first probe uses only this safe area.
# The contact sheet still requires visual review for central reticles/characters.
ROI = (155, 50, 1125, 595)


def main() -> None:
    root = configure_workspace()
    manifest = root / "data/processed/style-target-20261001-v14-expanded-reviewed/dataset-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("status") != "approved" or data.get("image_count") != len(data.get("records", [])):
        raise ValueError("Target source manifest is not approved")
    by_name = {row["file"]: row for row in data["records"]}
    if len(set(FILES)) != len(FILES):
        raise ValueError("Duplicate target image")
    rows = []
    contact = Image.new("RGB", (4 * 480, 3 * 300), "#151515")
    draw = ImageDraw.Draw(contact)
    for n, name in enumerate(FILES):
        source = by_name.get(name)
        if source is None:
            raise ValueError(f"Target absent from approved source: {name}")
        path = manifest.parent / name
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"Target changed: {name}")
        image = Image.open(path).convert("RGB")
        if image.size != (1280, 720):
            raise ValueError(f"Unexpected size: {name}")
        crop = image.crop(ROI)
        thumb = ImageOps.fit(crop, (480, 270), method=Image.Resampling.LANCZOS)
        x, y = (n % 4) * 480, (n // 4) * 300
        contact.paste(thumb, (x, y + 25))
        draw.text((x + 5, y + 5), name, fill="white")
        rows.append({"file": name, "sha256": source["sha256"],
                     "source_video": source.get("source_video"),
                     "second": source.get("second"), "safe_roi_xyxy": list(ROI)})
    out = root / "runs/quiet-clip-style-audit-20261002/focused-targets"
    out.mkdir(parents=True, exist_ok=True)
    board = out / "focused_urban11_safe_crops.jpg"
    contact.save(board, quality=95)
    write_json(out / "target-selection.json", {
        "schema": "focused-urban-target-selection-v1",
        "status": "visual_review_required",
        "parent_manifest": str(manifest),
        "parent_manifest_sha256": file_hash(manifest),
        "image_count": len(rows), "records": rows,
        "policy": "Daylight urban walls, paving and stairs with worn concrete/brick; no panorama, night, sunset, or vehicle-dominated scenes. ROI removes edge HUD/weapon but requires central visual review.",
        "contact_sheet": str(board), "contact_sheet_sha256": file_hash(board),
    })
    print(board)


if __name__ == "__main__":
    main()
