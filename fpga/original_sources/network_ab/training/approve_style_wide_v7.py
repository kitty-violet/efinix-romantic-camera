"""Create an immutable, visually reviewed panorama expansion from the v7 draft."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from .data import audit_style
from .derive_style_wide_v7 import contact
from .runtime import configure_workspace, file_hash, write_json


KEEP_NEW = {
    "W_N_2_00970.png": "sharp indoor mesh and concrete, no overlay",
    "W_N_2_00980.png": "sharp indoor corridor, no overlay",
    "W_N_3_00205.png": "clear outdoor crane and panoramic city, no overlay",
    "W_N_3_00550.png": "clear concrete construction material, no overlay",
    "W_N_3_00560.png": "clear wide construction view, no overlay",
    "W_N_3_00565.png": "clear crane and masonry view, no overlay",
    "W_N_3_00570.png": "clear open construction panorama, no overlay",
}
REJECT_NEW = {
    "W_N_2_00945.png": "large blood smear",
    "W_N_2_00975.png": "large foreground human and near-duplicate indoor view",
    "W_N_3_00215.png": "near-duplicate crane panorama at 00205",
    "W_N_3_00220.png": "motion softness and near-duplicate crane view",
    "W_N_3_01200.png": "large red foreground vehicle and right-edge UI",
    "W_N_3_01205.png": "large red foreground vehicle and right-edge UI",
    "W_N_3_01210.png": "blur and dark edge splatter",
    "W_N_3_01220.png": "right-edge weapon and UI",
    "W_N_3_01235.png": "right-edge weapon, UI, and softness",
}


def main():
    root = configure_workspace() / "data/processed"
    draft_dir = root / "style-target-20261001-v7-wide-expanded-draft"
    draft_path = draft_dir / "dataset-manifest.draft.json"
    final_dir = root / "style-target-20261001-v7-wide-reviewed"
    if final_dir.exists():
        raise FileExistsError(final_dir)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    new = {r["file"] for r in draft["records"] if r["file"].startswith("W_N_")}
    if new != KEEP_NEW.keys() | REJECT_NEW.keys() or len(draft["records"]) != 41:
        raise ValueError("Draft candidate set changed since visual review")
    final_dir.mkdir(parents=True)
    (final_dir / "review").mkdir()
    rows = []
    for original in draft["records"]:
        if original["file"] in REJECT_NEW:
            continue
        source = draft_dir / original["file"]
        if file_hash(source) != original["sha256"]:
            raise ValueError(f"Source changed: {source}")
        row = copy.deepcopy(original)
        if row["file"] in KEEP_NEW:
            row["review"] = KEEP_NEW[row["file"]]
        shutil.copy2(source, final_dir / row["file"])
        rows.append(row)
    wide = sum(r["kind"] == "wide_scene" for r in rows)
    textures = sum(r["kind"] == "texture_region" for r in rows)
    if (len(rows), wide, textures) != (32, 12, 20):
        raise ValueError("Unexpected final target counts")
    pages = contact(final_dir, rows, "wide_scene") + contact(final_dir, rows, "texture_region")
    manifest = copy.deepcopy(draft)
    manifest.update(status="approved", records=rows, image_count=len(rows),
        wide_scene_count=wide, texture_region_count=textures,
        source_times=sorted({f"{r['source_video']}:{r['requested_second']}" for r in rows}),
        contact_pages=pages, draft_manifest=str(draft_path),
        draft_manifest_sha256=file_hash(draft_path),
        visual_review={"kept_new": KEEP_NEW, "rejected_new": REJECT_NEW,
            "reason": "Native resolution contact and suspect full frames inspected; near-focus blur, UI, blood, foreground occlusion, and duplicates removed",
            "contact_sha256": {str(p): file_hash(p) for p in map(Path, pages)}})
    write_json(final_dir / "dataset-manifest.json", manifest)
    audit_style(final_dir)
    print(json.dumps({"folder": str(final_dir), "count": len(rows),
                      "wide": wide, "textures": textures, "contacts": pages}, indent=2))


if __name__ == "__main__":
    main()
