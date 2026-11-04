"""Build a wider, cleaner daylight game target candidate for visual review."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from .data import audit_style
from .runtime import configure_workspace, file_hash, write_json


NEIGHBORS = {
    "2_00945", "2_00970", "2_00975", "2_00980",
    "3_00205", "3_00215", "3_00220",
    "3_00550", "3_00560", "3_00565", "3_00570",
    "3_01200", "3_01205", "3_01210", "3_01220", "3_01235",
}
EXCLUDED_PARENT = {"W_2_00495.png": "blood-heavy interior",
                   "T_1_00065_A.png": "blood-heavy wall"}


def contact(folder, rows, kind):
    selected = [r for r in rows if r["kind"] == kind]
    pages = []
    for first in range(0, len(selected), 24):
        chunk = selected[first:first + 24]
        canvas = Image.new("RGB", (4 * 330, ((len(chunk) + 3) // 4) * 210), "#101010")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(chunk):
            x, y = i % 4 * 330, i // 4 * 210
            with Image.open(folder / row["file"]) as image:
                canvas.paste(image.resize((330, 185)), (x, y + 22))
            draw.text((x + 3, y + 3), row["file"], fill="white")
        path = folder / "review" / f"contact_{kind}_{first//24:02d}.jpg"
        canvas.save(path, quality=94)
        pages.append(str(path))
    return pages


def main():
    root = configure_workspace()
    parent = root / "data/processed/style-target-20261001-v6-daylight-draft"
    parent_manifest = parent / "dataset-manifest.json"
    old = json.loads(parent_manifest.read_text(encoding="utf-8"))
    audit_style(parent)
    audit_path = root / "runs/style-neighbor-audit-20261001/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = {Path(r["path"]).stem: r for r in audit["candidates"]}
    if not NEIGHBORS <= candidates.keys():
        raise ValueError("Missing reviewed native candidate")
    folder = root / "data/processed/style-target-20261001-v7-wide-expanded-draft"
    if folder.exists():
        raise FileExistsError(folder)
    (folder / "review").mkdir(parents=True)
    rows = []
    for original in old["records"]:
        if original["file"] in EXCLUDED_PARENT:
            continue
        row = copy.deepcopy(original)
        shutil.copy2(parent / row["file"], folder / row["file"])
        rows.append(row)
    for key in sorted(NEIGHBORS):
        candidate = candidates[key]
        source = Path(candidate["path"])
        if file_hash(source) != candidate["sha256"]:
            raise ValueError("Candidate image changed")
        name = f"W_N_{key}.png"
        shutil.copy2(source, folder / name)
        rows.append({"file": name, "sha256": file_hash(folder / name),
            "role": "style_only", "kind": "wide_scene", "width": 1320, "height": 742,
            "source_video": Path(candidate["source"]).name,
            "source_sha256": candidate["source_sha256"],
            "requested_second": candidate["requested_second"],
            "crop_xywh_native": candidate["crop_xywh_native"],
            "reviewed_source_sha256": candidate["sha256"],
            "review": "native neighboring wide candidate; final visual review pending"})
    if len(rows) != 41 or sum(r["kind"] == "wide_scene" for r in rows) != 21:
        raise ValueError("Unexpected candidate coverage")
    contacts = contact(folder, rows, "wide_scene") + contact(folder, rows, "texture_region")
    manifest = copy.deepcopy(old)
    manifest.update(status="draft_requires_visual_review", image_count=len(rows),
        wide_scene_count=21, texture_region_count=20, records=rows,
        contact_pages=contacts, parent_manifest=str(parent_manifest),
        parent_manifest_sha256=file_hash(parent_manifest),
        neighbor_audit=str(audit_path), neighbor_audit_sha256=file_hash(audit_path),
        selection_policy="More native daylight panoramas, no near-field blur/UI or blood-heavy parent tiles; manual final review required",
        excluded_parent=EXCLUDED_PARENT)
    write_json(folder / "dataset-manifest.draft.json", manifest)
    print(json.dumps({"folder": str(folder), "images": len(rows),
                      "wide": 21, "texture": 20, "contacts": contacts}, indent=2))


if __name__ == "__main__":
    main()
