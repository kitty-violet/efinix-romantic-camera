"""Approve reviewed daylight-only target ablation without changing v5."""
from __future__ import annotations

import json
from pathlib import Path

from .data import audit_style
from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace() / "data/processed"
    folder = root / "style-target-20261001-v6-daylight-draft"
    draft = folder / "dataset-manifest.draft.json"
    approved = folder / "dataset-manifest.json"
    if approved.exists():
        raise FileExistsError(approved)
    manifest = json.loads(draft.read_text(encoding="utf-8"))
    if manifest["status"] != "draft_requires_visual_review" or len(manifest["records"]) != 27:
        raise ValueError("Unexpected target draft")
    review = folder / "review"
    review.mkdir(exist_ok=True)
    contacts = []
    for kind in ("wide_scene", "texture_region"):
        source = folder / f"contact_{kind}.jpg"
        target = review / source.name
        if source.exists():
            source.rename(target)
        if not target.is_file():
            raise FileNotFoundError(target)
        contacts.append({"path": str(target), "sha256": file_hash(target)})
    records = manifest["records"]
    counts = {kind: sum(r["kind"] == kind for r in records)
              for kind in ("wide_scene", "texture_region")}
    if counts != {"wide_scene": 6, "texture_region": 21}:
        raise ValueError("Daylight subset count changed")
    manifest.update(status="approved", wide_scene_count=6, texture_region_count=21,
        source_times=sorted({f"{r['source_video']}:{r['requested_second']}" for r in records}),
        contact_pages=[r["path"] for r in contacts],
        daylight_review={"role": "experimental ablation only; original v5 unchanged",
            "visual_review": "all six wide and 21 native texture crops inspected in two contact sheets",
            "contact_hashes": contacts,
            "excluded_late_dusk": "3.mp4 times later than 1880s",
            "retained_low_ui_native_images": True})
    write_json(approved, manifest)
    audit_style(folder)
    print(json.dumps({"approved": str(approved), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
