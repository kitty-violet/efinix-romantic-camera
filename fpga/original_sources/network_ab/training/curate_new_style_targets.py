"""Review and optionally approve three new game style videos into v14.

Default mode creates a review-only selection, with explicit provenance and
rejections. --finalize copies the selected images plus approved v13 into a
new, immutable target directory after the contacts have been inspected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from .review_new_style_sources import sha256

ROOT = Path(r"E:\fpga\model-training-workspace")
RUN = ROOT / "runs/style-target-new-sources-20261001"
CANDIDATES = RUN / "native-candidates/candidate-manifest.json"
LATEST = RUN / "latest-video-review/candidate-manifest.json"
REVIEW = RUN / "curation-review-v2"
PARENT = ROOT / "data/processed/style-target-20261001-v13-final-reviewed"
FINAL = ROOT / "data/processed/style-target-20261001-v14-expanded-reviewed"

ACCEPTED = {
    "18-54-23": [110, 115, 125, 160],
    "18-59-02": [20, 35, 55, 130, 175, 205, 235,
                  605, 625, 645, 650, 660, 690, 695, 700,
                  755, 765, 770, 840, 845, 855, 1045, 1070,
                  1125, 1130, 1985, 2065, 2080],
    "latest": [7, 11, 15, 35],
}
PANORAMAS = {
    ("18-54-23", 115),
    ("18-59-02", 235), ("18-59-02", 650), ("18-59-02", 660),
    ("18-59-02", 840), ("18-59-02", 845), ("18-59-02", 1125),
    ("18-59-02", 1985), ("18-59-02", 2065),
    ("latest", 7), ("latest", 11), ("latest", 15),
}
LOW_LIGHT = {("18-59-02", 1455), ("18-59-02", 1615),
             ("18-59-02", 1985), ("18-59-02", 2065)}

# The first two clips were checked chronologically at 5 s intervals and at a
# native crop; the last one at 2 s intervals. Reasons correspond to contact
# sheet inspection, not to a sharpness threshold alone.
REJECTION_GROUPS = {
    "foreground_bar_and_repeated_view": {("18-54-23", t) for t in [75, 80, 85, 90]},
    "motion_blur": {("18-54-23", 120), ("18-54-23", 155),
                    ("18-59-02", 215),
                    ("18-59-02", 1465), ("18-59-02", 1500),
                    ("18-59-02", 2070)},
    "near_field_or_dominant_foreground": {
        ("18-59-02", t) for t in [10, 45, 65, 105, 110, 120, 125,
                                   180, 260, 270, 280, 615, 665, 675,
                                   680, 750, 780, 850, 1050, 1090,
                                   1150, 1165, 1220, 1685, 1705]},
    "weak_texture_or_backlight": {("18-54-23", 150),
                                  ("18-59-02", 1455),
                                  ("18-59-02", 1615),
                                  ("18-59-02", 1485), ("18-59-02", 2075),
                                  ("latest", 9), ("latest", 13),
                                  ("latest", 21), ("latest", 23)},
    "repeated_camera_angle": {("18-59-02", 240), ("latest", 1),
                              ("latest", 3), ("latest", 17),
                              ("latest", 25), ("latest", 27),
                              ("latest", 33)},
    "game_ui_or_desktop": {("latest", 37)},
}


def source_rows() -> list[dict]:
    first = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    second = json.loads(LATEST.read_text(encoding="utf-8"))
    if first["status"] != "DRAFT_NOT_FOR_TRAINING" or second["status"] != "DRAFT_NOT_FOR_TRAINING":
        raise ValueError("Candidate review state changed")
    rows = []
    for row in first["rows"]:
        rows.append({**row, "tag": row["source_tag"],
                     "candidate_path": str(CANDIDATES.parent / row["file"])})
    for row in second["rows"]:
        rows.append({**row, "tag": "latest",
                     "source_file": second["source"]["file"],
                     "candidate_path": str(LATEST.parent / row["file"])})
    return rows


def decision(row: dict) -> tuple[str, str]:
    key = row["tag"], row["second"]
    if row["second"] in ACCEPTED[row["tag"]]:
        return "accepted", "clear full-scene game texture; limited UI/foreground after native crop"
    for reason, members in REJECTION_GROUPS.items():
        if key in members:
            return "rejected", reason
    return "rejected", "lower_scene_value_or_redundant_relative_to_selected"


def contacts(rows: list[dict]) -> list[str]:
    pages = []
    for first in range(0, len(rows), 16):
        canvas = Image.new("RGB", (1680, 1072), "#171717")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[first:first + 16]):
            x, y = slot % 4 * 420, slot // 4 * 268
            with Image.open(row["candidate_path"]) as image:
                canvas.paste(image.resize((420, 236), Image.Resampling.LANCZOS), (x, y + 28))
            label = f"{row['tag']} {row['second']}s {row['scene_scale']}"
            draw.text((x + 5, y + 5), label, fill="white")
        path = REVIEW / f"accepted_{first // 16:02d}.jpg"
        canvas.save(path, quality=94)
        pages.append(str(path))
    return pages


def review() -> None:
    if REVIEW.exists():
        raise FileExistsError(REVIEW)
    REVIEW.mkdir(parents=True)
    rows = source_rows()
    expected = sum(map(len, ACCEPTED.values()))
    keys = {(r["tag"], r["second"]) for r in rows}
    if len(keys) != len(rows) or any((tag, t) not in keys for tag, times in ACCEPTED.items() for t in times):
        raise ValueError("Candidate rows do not match selection")
    selected = []
    for row in rows:
        status, reason = decision(row)
        row["decision"] = status
        row["reason"] = reason
        row["scene_scale"] = "panorama" if (row["tag"], row["second"]) in PANORAMAS else "full_scene"
        row["lighting"] = "dusk_or_night" if (row["tag"], row["second"]) in LOW_LIGHT else "daylight"
        if status == "accepted":
            selected.append(row)
    if len(selected) != expected:
        raise ValueError("Selected count mismatch")
    pages = contacts(selected)
    report = {"schema": "new-style-visual-selection-v1",
              "status": "REVIEW_ONLY", "sampled_candidate_count": len(rows),
              "accepted_count": len(selected), "rejected_count": len(rows) - len(selected),
              "panorama_count": sum(r["scene_scale"] == "panorama" for r in selected),
              "lighting_counts": {name: sum(r["lighting"] == name for r in selected)
                                  for name in ["daylight", "dusk_or_night"]},
              "source_counts": {tag: sum(r["tag"] == tag and r["decision"] == "accepted" for r in rows)
                                for tag in ACCEPTED},
              "crop_xywh_native": [0, 125, 1280, 720],
              "notes": "Visual screening at native crop; reject menus/desktop, motion blur, near-field dominance and repeat views. Intentional game atmosphere and a few narrow edge weapon remnants are recorded; no inpainting or rescaling.",
              "rows": rows, "accepted_contact_pages": pages}
    (REVIEW / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reviewed={len(rows)} accepted={len(selected)} panorama={report['panorama_count']}")


def finalize() -> None:
    if FINAL.exists():
        raise FileExistsError(FINAL)
    selection = json.loads((REVIEW / "selection.json").read_text(encoding="utf-8"))
    if selection["status"] != "REVIEW_ONLY":
        raise ValueError("Review status mismatch")
    parent_manifest_path = PARENT / "dataset-manifest.json"
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent["status"] != "approved":
        raise ValueError("Parent style target is not approved")
    accepted = [r for r in selection["rows"] if r["decision"] == "accepted"]
    if len(accepted) != selection["accepted_count"]:
        raise ValueError("Review count changed")
    FINAL.mkdir(parents=True)
    records = []
    for record in parent["records"]:
        source = PARENT / record["file"]
        if sha256(source) != record["sha256"]:
            raise ValueError(f"Parent file changed: {source}")
        shutil.copy2(source, FINAL / source.name)
        records.append(record)
    for row in accepted:
        source = Path(row["candidate_path"])
        if sha256(source) != row["sha256"]:
            raise ValueError(f"Candidate file changed: {source}")
        dest = FINAL / row["file"]
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copy2(source, dest)
        records.append({"file": dest.name, "sha256": row["sha256"],
                        "role": "style_only", "kind": "wide_scene",
                        "width": 1280, "height": 720,
                        "source_video": Path(row["source_file"]).name,
                        "source_sha256": row["source_sha256"],
                        "second": row["second"],
                        "scene": row["scene_scale"], "lighting": row["lighting"],
                        "crop_xywh_native": row["crop_xywh_native"],
                        "laplacian_variance_native": row["laplacian_variance_native"],
                        "luma_mean": row["luma_mean"],
                        "review": row["reason"]})
    manifest = {"schema": "model-training-style-v1", "status": "approved",
                "image_count": len(records), "wide_scene_count": len(records),
                "texture_region_count": 0,
                "sampling_policy": {"wide_scene_probability": 1.0,
                                    "texture_region_probability": 0.0},
                "source_videos": sorted({r["source_video"] for r in records}),
                "source_times": [f"{r['source_video']}:{r['second']}" for r in records],
                "selection_policy": "Parent v13 approved plus 3 newly screened videos; native 1280x720 scenes, motion blur/UI/near-field/repeats removed; no inpainting or rescaling",
                "source_scan": str(RUN / "timeline.json"),
                "latest_source_review": str(LATEST),
                "new_visual_review": str(REVIEW / "selection.json"),
                "parent_approved": str(parent_manifest_path),
                "parent_approved_sha256": sha256(parent_manifest_path),
                "new_image_count": len(accepted),
                "new_panorama_count": selection["panorama_count"],
                "records": records}
    (FINAL / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"final target: {len(records)} images ({len(accepted)} new)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    finalize() if args.finalize else review()


if __name__ == "__main__":
    main()
