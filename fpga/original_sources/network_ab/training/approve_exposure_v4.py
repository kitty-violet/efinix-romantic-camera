"""Approve the visually reviewed -1 EV experimental content set.

The seven train/val/test contact pages must be inspected before running this
one-off audit. Earlier approved masters and their manifests remain untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_content_manifest
from .media import HLG_TO_SRGB
from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace() / "data/processed"
    old_path = root / "content-20261001-v3-normal-validation/manifest.approved.json"
    draft_path = root / "content-20261001-v4-minus1ev-draft/manifest.draft.json"
    approved_path = draft_path.with_name("manifest.approved.json")
    if approved_path.exists():
        raise FileExistsError(approved_path)
    old = load_content_manifest(old_path)
    new = json.loads(draft_path.read_text(encoding="utf-8"))
    if new["status"] != "draft_requires_visual_review" or new["hlg_exposure_stops"] != -1:
        raise ValueError("Unexpected draft policy")
    if len(old["images"]) != 242 or len(new["images"]) != 242:
        raise ValueError("Selection count changed")
    for a, b in zip(old["images"], new["images"]):
        for field in ("source", "source_sha256", "frame_index", "split", "group", "timestamp"):
            if a[field] != b[field]:
                raise ValueError(f"Selection drift: {field}")
        if file_hash(b["path"]) != b["sha256"] or b["parent_image_sha256"] != a["sha256"]:
            raise ValueError("New/old image hash mismatch")
        b["effective_rgb_filter"] = HLG_TO_SRGB.replace(
            "tonemap=tonemap=mobius", "exposure=exposure=-1,tonemap=tonemap=mobius")
    review_dir = draft_path.parent / "review"
    contacts = sorted(review_dir.rglob("contact_*.jpg"))
    if len(contacts) != 7:
        raise ValueError("Seven visually reviewed split contact sheets required")
    stats = {}
    for split in ("train", "val", "test"):
        before = [r["bright_fraction"] for r in old["images"] if r["split"] == split]
        after = [r["bright_fraction"] for r in new["images"] if r["split"] == split]
        if len(before) != len(after) or np.mean(after) >= np.mean(before):
            raise ValueError(f"Exposure improvement failed: {split}")
        stats[split] = {"count": len(after), "mean_clipped_before": float(np.mean(before)),
                        "mean_clipped_after": float(np.mean(after))}
    new.update(status="approved", exposure_review={
        "scope": "experimental minus-one-EV decode of the same 242 selected frames",
        "contact_sheets": [{"path": str(p), "sha256": file_hash(p)} for p in contacts],
        "visual_review": "all seven train/val/test sheets inspected; same scenes and composition; less white clipping, some source highlights still bright",
        "paired_statistics": stats, "old_masters_untouched": True,
        "test_policy": "source-disjoint test images never sampled by CUT training or validation"})
    write_json(approved_path, new)
    load_content_manifest(approved_path)
    print(json.dumps({"approved": str(approved_path), "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
