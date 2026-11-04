"""Add visually reviewed gap discoveries to the clean wide-scene target set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

ROOT = Path(r"E:\fpga\model-training-workspace")
OLD = ROOT / "data/processed/style-target-20261001-v10-quality-reviewed"
GAP = ROOT / "data/processed/style-target-20261001-v11-gap-scout-draft"
OUT = ROOT / "data/processed/style-target-20261001-v12-quality-reviewed"
EXTRA = {"2.mp4": [410, 1080],
         "3.mp4": [1250, 1375, 2250, 2260, 2275, 2720, 3335]}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    old_path = OLD / "dataset-manifest.json"
    gap_path = GAP / "review/candidate-manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    gap = json.loads(gap_path.read_text(encoding="utf-8"))
    if old["status"] != "approved" or gap["status"] != "DRAFT_NOT_FOR_TRAINING":
        raise ValueError("Wrong parent status")
    selected = {(video, second) for video, times in EXTRA.items() for second in times}
    if not selected <= {(r["video"], r["second"]) for r in gap["rows"]}:
        raise ValueError("Gap selection is missing")
    (OUT / "review").mkdir(parents=True)
    records = []
    for row in old["records"]:
        source = OLD / row["file"]
        dest = OUT / row["file"]
        if sha(source) != row["sha256"]:
            raise ValueError(source)
        shutil.copy2(source, dest)
        records.append(row)
    decisions = []
    for row in gap["rows"]:
        key = (row["video"], row["second"])
        if key in selected:
            source = GAP / row["file"]
            dest = OUT / row["file"]
            if sha(source) != row["sha256"]:
                raise ValueError(source)
            shutil.copy2(source, dest)
            records.append({"file": row["file"], "sha256": row["sha256"],
                            "role": "style_only", "kind": "wide_scene",
                            "width": 1280, "height": 720,
                            "source_video": row["video"], "source_sha256": row["source_sha256"],
                            "second": row["second"], "scene": row["scene"],
                            "crop_xywh_native": row["crop_xywh_native"],
                            "laplacian_variance_native": row["laplacian_variance_native"],
                            "luma_mean": row["luma_mean"],
                            "review": "Visually inspected all v11 gap contact sheets; distinct clear game scene"})
        decisions.append({"source_video": row["video"], "second": row["second"],
                          "decision": "accept" if key in selected else "reject",
                          "reason": "distinct_clear_game_scene" if key in selected else "menu_cutscene_blur_foreground_or_less_representative"})
    records.sort(key=lambda r: (r["source_video"], r["second"]))
    pages = []
    for first in range(0, len(records), 12):
        canvas = Image.new("RGB", (1536, 1288), "#151515")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(records[first:first + 12]):
            x, y = index % 3 * 512, index // 3 * 322
            with Image.open(OUT / row["file"]) as source:
                if source.size != (1280, 720):
                    raise ValueError(row["file"])
                canvas.paste(source.resize((512, 288), Image.Resampling.LANCZOS), (x, y + 30))
            draw.text((x + 5, y + 6), f"{row['source_video']} {row['second']}s {row['scene']}", fill="white")
        target = OUT / "review" / f"approved_{first // 12:02d}.jpg"
        canvas.save(target, quality=94)
        pages.append(str(target))
    report = {"schema": "model-training-style-v1", "status": "approved",
              "image_count": len(records), "wide_scene_count": len(records),
              "texture_region_count": 0, "source_videos": sorted({r["source_video"] for r in records}),
              "source_times": [f"{r['source_video']}:{r['second']}" for r in records],
              "selection_policy": "Visual approval across full-timeline first pass and omitted-span gap scan; native 1280x720 only; no menus or temporary game filters",
              "source_scan": old["source_scan"],
              "parent_approved": str(old_path), "parent_approved_sha256": sha(old_path),
              "gap_draft": str(gap_path), "gap_draft_sha256": sha(gap_path),
              "records": records, "contact_pages": pages,
              "visual_review": "All 9 initial and 7 gap contact sheets inspected 2026-10-01"}
    (OUT / "dataset-manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "review/gap-decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"approved={len(records)} new={len(selected)} rejected_gap={len(decisions)-len(selected)}")


if __name__ == "__main__":
    main()
