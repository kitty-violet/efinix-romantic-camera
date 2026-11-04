"""Build a source-disjoint, full-scene evaluation index from approved HLG masters.

The primary test excludes hand-focused frames; those remain a separate hand
stress test. Existing train images and their approved manifest are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw

ROOT = Path(r"E:\fpga\model-training-workspace")
PARENT = ROOT / "data/processed/content-20261001-v4-minus1ev-draft/manifest.approved.json"
OUT = ROOT / "runs/content-quality-eval-20261001"
PICKS = {
    "primary_test": {
        "VID_20260925_070032.mp4": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39],
        "VID_20260926_192624.mp4": [0, 3, 6, 9, 12],
    },
    "hand_near_field_stress": {"VID_20260925_070032.mp4": [42, 43, 44, 45, 46]},
    "validation_scene_coverage": {
        "VID_20260925_095525.mp4": [0, 4, 8, 12, 18, 20, 24, 28, 32, 36, 40],
        "VID_20260925_101046.mp4": [0, 4, 8, 13, 16, 20],
    },
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contact(rows: list[dict], name: str) -> str:
    width = 3 * 480
    nrows = (len(rows) + 2) // 3
    image = Image.new("RGB", (width, nrows * 302), "#171717")
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows):
        x, y = index % 3 * 480, index // 3 * 302
        with Image.open(row["path"]) as source:
            image.paste(source.resize((480, 270), Image.Resampling.LANCZOS), (x, y + 28))
        draw.text((x + 5, y + 5),
                  f"{Path(row['source']).stem[-6:]} {row['timestamp']:.1f}s L{row['laplacian_variance']:.0f}",
                  fill="white")
    dest = OUT / f"{name}.jpg"
    image.save(dest, quality=94)
    return str(dest)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if parent["status"] != "approved":
        raise ValueError("Parent dataset not approved")
    picked = {}
    images = parent["images"]
    for role, sources in PICKS.items():
        rows = []
        for source, seconds in sources.items():
            eligible = [r for r in images if Path(r["source"]).name == source]
            if not eligible:
                raise ValueError(f"No approved images for {source}")
            expected_split = "val" if role == "validation_scene_coverage" else "test"
            if {r["split"] for r in eligible} != {expected_split}:
                raise ValueError(f"Wrong split: {source}")
            for second in seconds:
                best = min(eligible, key=lambda r: abs(r["timestamp"] - second))
                if abs(best["timestamp"] - second) > 0.6:
                    raise ValueError(f"Missing timestamp near {source}:{second}")
                if best not in rows:
                    rows.append(best)
        for row in rows:
            path = Path(row["path"])
            if sha(path) != row["sha256"]:
                raise ValueError(f"Image hash changed: {path}")
        picked[role] = rows
    if {r["path"] for r in picked["primary_test"]} & {r["path"] for r in picked["hand_near_field_stress"]}:
        raise ValueError("Primary and near-field test overlap")
    OUT.mkdir(parents=True)
    reports = {}
    for role, rows in picked.items():
        reports[role] = {"count": len(rows),
                         "median_laplacian": median(r["laplacian_variance"] for r in rows),
                         "source_videos": sorted({Path(r["source"]).name for r in rows}),
                         "paths": [r["path"] for r in rows],
                         "contact_sheet": contact(rows, role)}
    report = {"schema": "content-evaluation-v5", "status": "reviewed_evaluation_index",
              "parent_manifest": str(PARENT), "parent_sha256": sha(PARENT),
              "purpose": "Whole-scene primary validation/test; hand focus shift is a separate diagnostic, not the showcase target",
              "training_data_unchanged": True, "splits_source_disjoint": True,
              "selections": reports}
    (OUT / "index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: value["count"] for key, value in reports.items()})


if __name__ == "__main__":
    main()
