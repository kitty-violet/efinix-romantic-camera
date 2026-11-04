"""Create a source-disjoint, sharp normal-scene validation manifest."""

import copy
import json
from pathlib import Path

from network_ab.training.data import load_content_manifest
from network_ab.training.runtime import canonical_hash, configure_workspace, file_hash, write_json

ROOT = configure_workspace()
SOURCE = ROOT / "data/processed/content-20260930-v2/manifest.approved.json"
TARGET = ROOT / "data/processed/content-20261001-v3-normal-validation"
VALIDATION_SOURCES = {"VID_20260925_095525.mp4", "VID_20260925_101046.mp4"}


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    old = load_content_manifest(SOURCE)
    images, diagnostics, exclusions = [], [], []
    for original in old["images"]:
        row = copy.deepcopy(original)
        name = Path(row["source"]).name
        if row["split"] == "val":
            diagnostics.append(row)
            continue
        if name in VALIDATION_SOURCES:
            row["split"] = "val"
            row["group"] = "normal-park-path-plaza-validation"
        # Texture statistics are a candidate filter, followed by earlier visual review.
        if row["split"] != "test" and (row["laplacian_variance"] < 300
                                             or row["bright_fraction"] > 0.35):
            exclusions.append({"path": row["path"], "reason": "low edge detail or highlight clipping",
                               "laplacian_variance": row["laplacian_variance"],
                               "bright_fraction": row["bright_fraction"]})
            continue
        images.append(row)
    val_sources = {row["source"]: row for row in images if row["split"] == "val"}
    videos = [copy.deepcopy(v) for v in old["videos"] if v["split"] == "test"]
    for source, row in sorted(val_sources.items()):
        videos.append({"path": source, "sha256": row["source_sha256"],
                       "group": "normal-park-path-plaza-validation", "split": "val",
                       "decode": row["source_decode"]})
    for video in videos:
        if file_hash(video["path"]) != video["sha256"]:
            raise ValueError(f"Source video changed: {video['path']}")
    counts = {split: sum(r["split"] == split for r in images)
              for split in ("train", "val", "test")}
    if min(counts.values()) == 0:
        raise ValueError(f"Empty split: {counts}")
    showcase = [r["path"] for r in images if r["split"] == "test"
                and r["laplacian_variance"] >= 300 and r["bright_fraction"] <= 0.3]
    manifest = copy.deepcopy(old)
    manifest.update(
        status="approved", images=images, videos=videos,
        plan_sha256=canonical_hash({"parent": file_hash(SOURCE),
                                    "val_sources": sorted(VALIDATION_SOURCES),
                                    "quality": {"min_laplacian_variance": 300,
                                                "max_bright_fraction": 0.35}}),
        parent_manifest=str(SOURCE), parent_manifest_sha256=file_hash(SOURCE),
        correction="Normal-focus park/path/plaza source videos moved entirely to validation; soft near-field corridor held as diagnostic; train excludes low-detail/clipped frames. Original decoded images unchanged.",
        validation_policy="Source-disjoint normal-focus validation; test remains untouched and unseen",
        selection_policy="Source-level split, 300 Laplacian threshold and 0.35 clipped-bright fraction for train/val only; manual contact-sheet review",
        diagnostic_near_field={"count": len(diagnostics),
                               "sources": sorted({r["source"] for r in diagnostics}),
                               "images": [r["path"] for r in diagnostics]},
        quality_exclusions=exclusions,
        showcase_test_paths=showcase,
        review={"method": "all-group contact sheets plus source and focus audit",
                "source_splits_unchanged": False,
                "test_images_used_for_training": False,
                "counts": counts,
                "originals_untouched": True})
    TARGET.mkdir(parents=True)
    write_json(TARGET / "manifest.approved.json", manifest)
    load_content_manifest(TARGET / "manifest.approved.json")
    print(json.dumps({"counts": counts, "diagnostic_near_field": len(diagnostics),
                      "quality_exclusions": len(exclusions),
                      "sharp_test_showcase": len(showcase)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
