"""Measure sampled game palettes without declaring one universal game tint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .runtime import file_hash, write_json


def image_stats(image: np.ndarray) -> dict:
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return {"median_lab_l": float(np.median(lab[:, :, 0])),
            "median_lab_b": float(np.median(lab[:, :, 2])),
            "median_hsv_s": float(np.median(hsv[:, :, 1])),
            "dark_fraction_l_below_80": float((lab[:, :, 0] < 80).mean()),
            "blue_fraction_lab_b_below_120": float((lab[:, :, 2] < 120).mean())}


def summarize(records: list[dict]) -> dict:
    return {key: {"median": float(np.median([r[key] for r in records])),
                  "range": [float(min(r[key] for r in records)),
                            float(max(r[key] for r in records))]}
            for key in records[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdoor-selection", type=Path, required=True)
    parser.add_argument("--indoor-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outdoor = json.loads(args.outdoor_selection.read_text(encoding="utf-8"))
    indoor = json.loads(args.indoor_candidates.read_text(encoding="utf-8"))
    if (outdoor["status"] != "reviewed_focused_urban" or
            indoor["status"] != "visual_review_required"):
        raise ValueError("Unexpected review status")
    manifest = Path(outdoor["parent_manifest"])
    if file_hash(manifest) != outdoor["parent_manifest_sha256"]:
        raise ValueError("Outdoor target manifest changed")
    outdoor_values = []
    for row in outdoor["records"]:
        path = manifest.parent / row["file"]
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Outdoor target changed: {path}")
        image = cv2.imread(str(path))
        x1, y1, x2, y2 = row["safe_roi_xyxy"]
        outdoor_values.append(image_stats(image[y1:y2, x1:x2]))
    indoor_values = []
    for row in indoor["records"]:
        path = Path(row["path"])
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Indoor target changed: {path}")
        indoor_values.append(image_stats(cv2.imread(str(path))))
    result = {"schema": "game-material-palette-audit-v1",
              "outdoor_source": str(args.outdoor_selection),
              "outdoor_sha256": file_hash(args.outdoor_selection),
              "outdoor_reviewed_count": len(outdoor_values),
              "outdoor_palette": summarize(outdoor_values),
              "indoor_source": str(args.indoor_candidates),
              "indoor_sha256": file_hash(args.indoor_candidates),
              "indoor_review_only_count": len(indoor_values),
              "indoor_palette": summarize(indoor_values),
              "interpretation": "Selection-biased image samples; indoor group contains effects, HUD and people. A warm/dark palette may fit a scene, never a universal hard rule."}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
