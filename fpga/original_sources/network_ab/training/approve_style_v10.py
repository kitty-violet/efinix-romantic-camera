"""Freeze a visually reviewed, diverse native-resolution target set.

Only inputs from the separately generated v9 left-crop draft are copied. Source
videos and older approved datasets remain immutable. No metric can approve a
frame on its own: APPROVED below is the record of contact-sheet visual review.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"E:\fpga\model-training-workspace")
SOURCE = ROOT / "data/processed/style-target-20261001-v9-leftcrop-draft"
DEST = ROOT / "data/processed/style-target-20261001-v10-quality-reviewed"

# Visually verified on all nine v9 contact sheets. Sparse dusk/night sampling
# prevents a sunset-orange bias; native 1280x720 scenes replace soft 512 crops.
APPROVED = {
    "2.mp4": [430, 455, 965],
    "3.mp4": [
        95, 115, 180, 375, 405, 550, 565, 620, 645, 655,
        905, 1035, 1095, 1150, 1160, 1205, 1245,
        1495, 1525, 1605, 1880, 1895, 1935, 2060,
        2380, 2410, 2435, 2480, 2560, 2570,
        2820, 2935, 3025, 3140, 3160, 3270, 3365,
        3380, 3580, 3645, 3775, 3810, 3830,
    ],
}

REJECT = {
    "desktop_or_game_menu": {
        ("2.mp4", x) for x in [5]
    } | {("3.mp4", x) for x in [760, 1530, 1770, 2970, 3215, 3655]},
    "temporary_in_game_visual_filter": {("3.mp4", x) for x in [2015, 2040]},
    "heavy_blur_or_soft_focus": {("2.mp4", x) for x in [270, 575, 990]}
    | {("3.mp4", x) for x in [80, 325, 410, 575, 690, 695, 720, 1190,
                                 1235, 1475, 2285, 2320, 2450, 2515,
                                 2600, 2845, 2850, 2940, 3405, 3490, 3530,
                                 3710, 3730, 3795]},
    "foreground_hand_weapon_ui_or_character_dominates": {
        ("2.mp4", x) for x in [40, 130, 145, 210, 240, 260, 595, 605,
                                 915, 925, 945]
    } | {("3.mp4", x) for x in [225, 235, 780, 1045, 1105, 1570, 1620,
                                 1640, 1665, 1675, 1755, 1790, 1805, 1915,
                                 1950, 2300, 2390, 2450, 2790, 2905, 2995,
                                 3050, 3170, 3190, 3300, 3455, 3525, 3600,
                                 3615, 3690]},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return sum(int(bit) << i for i, bit in enumerate((gray[:, 1:] > gray[:, :-1]).flat))


def create_contacts(records: list[dict]) -> list[str]:
    pages = []
    for first in range(0, len(records), 12):
        canvas = Image.new("RGB", (3 * 512, 4 * 322), "#151515")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(records[first:first + 12]):
            x, y = slot % 3 * 512, slot // 3 * 322
            with Image.open(DEST / row["file"]) as source:
                canvas.paste(source.resize((512, 288), Image.Resampling.LANCZOS), (x, y + 30))
            draw.text((x + 5, y + 6),
                      f"{row['source_video']} {row['second']}s {row['scene']} L{row['laplacian_variance_native']:.0f}",
                      fill="white")
        target = DEST / "review" / f"approved_{first // 12:02d}.jpg"
        canvas.save(target, quality=94)
        pages.append(str(target))
    return pages


def main() -> None:
    if DEST.exists():
        raise FileExistsError(DEST)
    draft_path = SOURCE / "review/candidate-manifest.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft["status"] != "DRAFT_NOT_FOR_TRAINING" or draft["crop_xywh_native"] != [0, 125, 1280, 720]:
        raise ValueError("Unexpected source draft")
    selected = {(video, second) for video, seconds in APPROVED.items() for second in seconds}
    available = {(row["video"], int(row["second"])) for row in draft["rows"]}
    if not selected <= available:
        raise ValueError(f"Selections absent from draft: {selected - available}")
    (DEST / "review").mkdir(parents=True)
    decisions = []
    records = []
    for row in draft["rows"]:
        key = (row["video"], int(row["second"]))
        reason = next((label for label, keys in REJECT.items() if key in keys),
                      "near_duplicate_or_less_representative_composition")
        if key in selected:
            reason = "accepted_visual_review"
            source_file = SOURCE / row["file"]
            if sha(source_file) != row["sha256"]:
                raise ValueError(f"Draft image hash mismatch: {source_file}")
            target_file = DEST / row["file"]
            shutil.copy2(source_file, target_file)
            if sha(target_file) != row["sha256"]:
                raise ValueError(f"Copied image hash mismatch: {target_file}")
            with Image.open(target_file) as image:
                if image.size != (1280, 720):
                    raise ValueError(f"Wrong size: {target_file}")
            records.append({"file": row["file"], "sha256": row["sha256"],
                            "role": "style_only", "kind": "wide_scene",
                            "width": 1280, "height": 720,
                            "source_video": row["video"], "source_sha256": row["source_sha256"],
                            "second": int(row["second"]), "scene": row["scene"],
                            "crop_xywh_native": row["crop_xywh_native"],
                            "laplacian_variance_native": row["laplacian_variance_native"],
                            "luma_mean": row["luma_mean"],
                            "dhash": f"{dhash(target_file):016x}",
                            "review": "Visually inspected v9 contact sheets; scene/texture sharpness and obstruction checked"})
        decisions.append({"source_video": row["video"], "second": int(row["second"]),
                          "decision": "accept" if key in selected else "reject", "reason": reason})
    if len(records) != len(selected):
        raise AssertionError("Selection count mismatch")
    # Flag visually similar frames for final human assessment; never delete silently.
    near = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            dist = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if dist <= 6:
                near.append({"a": a["file"], "b": b["file"], "dhash_distance": dist})
    report = {"schema": "model-training-style-v1", "status": "approved",
              "image_count": len(records), "wide_scene_count": len(records),
              "texture_region_count": 0,
              "source_videos": sorted(APPROVED),
              "source_times": [f"{r['source_video']}:{r['second']}" for r in records],
              "selection_policy": "Nine full contact sheets visually reviewed; preserve full native 1280x720 scenes, exclude menus, blur, obstruction and temporary filters; limit dusk/night redundancy",
              "source_scan": draft["source_scan"], "parent_draft": str(draft_path),
              "parent_draft_sha256": sha(draft_path),
              "records": records, "scene_counts": dict(Counter(r["scene"] for r in records)),
              "near_duplicate_flags": near,
              "contact_pages": create_contacts(records),
              "visual_review": "2026-10-01: all 132 candidates visually assessed; all selected images shown in approved contact pages"}
    (DEST / "dataset-manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEST / "review/decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accepted={len(records)} rejected={len(decisions)-len(records)} near_duplicate_flags={len(near)}")


if __name__ == "__main__":
    main()
