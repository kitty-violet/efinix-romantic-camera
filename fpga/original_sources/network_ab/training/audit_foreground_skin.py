"""Inspect how often the conservative training-only skin cue fires."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .foreground_skin import likely_foreground_skin
from .runtime import configure_workspace, write_json


def main():
    root = configure_workspace()
    data = json.loads((root / "data/processed/content-20260930-v2/manifest.approved.json").read_text(encoding="utf-8"))
    records = []
    for row in data["images"]:
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR)
            pixels = np.array(image, copy=True)
        value = torch.from_numpy(pixels).permute(2, 0, 1).float().unsqueeze(0).div(255)
        mask = likely_foreground_skin(value)[0, 0].numpy()
        records.append({"sha256": row["sha256"], "split": row["split"],
                        "count": int(mask.sum()), "fraction": float(mask.mean())})
        if row["split"] == "val" and row["sha256"] == [r for r in data["images"] if r["split"] == "val"][8]["sha256"]:
            overlay = pixels.copy()
            overlay[mask > 0] = .4*overlay[mask > 0] + .6*np.array([0, 255, 0])
            out = root / "runs/style-target-review-20261001/skin_mask_val008.png"
            Image.fromarray(overlay.astype("uint8")).save(out)
    counts = {split: {"images": sum(r["split"] == split for r in records),
                      "strong_mask_gt_1pct": sum(r["split"] == split and r["fraction"] > .01 for r in records)}
              for split in ("train", "val", "test")}
    write_json(root / "runs/style-target-review-20261001/skin_cue_audit.json",
               {"split_counts": counts, "records": records,
                "warning": "Weak cue only; verify overlay visually before using as loss"})
    print(counts)


if __name__ == "__main__":
    main()
