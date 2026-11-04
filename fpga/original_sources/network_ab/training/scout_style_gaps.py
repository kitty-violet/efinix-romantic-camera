"""Review omitted spans of the three uploaded game videos at native resolution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio_ffmpeg

from network_ab.training import curate_style_v8 as base

ROOT = Path(r"E:\fpga\model-training-workspace")
OUT = ROOT / "data/processed/style-target-20261001-v11-gap-scout-draft"
WINDOWS = {
    "2.mp4": [(280, 420, "indoor"), (610, 900, "indoor"),
              (990, 1180, "indoor")],
    "3.mp4": [(1250, 1470, "daylight-street"),
              (1680, 1870, "daylight-street"),
              (2080, 2280, "daylight-street"),
              (2620, 2780, "dusk-rooftop"),
              (3215, 3360, "dusk-rooftop")],
}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    scan = json.loads(base.SCAN.read_text(encoding="utf-8"))
    inventory = json.loads(base.INVENTORY.read_text(encoding="utf-8"))
    sources = {Path(r["file"]).name: r for r in inventory["sources"]}
    existing = json.loads((ROOT / "data/processed/style-target-20261001-v9-leftcrop-draft/review/candidate-manifest.json").read_text(encoding="utf-8"))
    seen = {(row["video"], int(row["second"])) for row in existing["rows"]}
    rows = []
    for video, windows in WINDOWS.items():
        if base.sha256(Path(sources[video]["file"])) != sources[video]["sha256"]:
            raise ValueError(f"Source changed: {video}")
        for lo, hi, scene in windows:
            for start in range(lo, hi, 15):
                eligible = [r for r in scan["rows"] if r["video"] == video
                            and start <= r["second"] < min(start + 15, hi)
                            and (video, int(r["second"])) not in seen]
                if not eligible:
                    continue
                best = max(eligible, key=lambda r: r["best_3_tile_laplacian"])
                rows.append({"video": video, "second": int(best["second"]),
                             "scene": scene,
                             "scan_best3_laplacian": best["best_3_tile_laplacian"]})
    base.OUT = OUT
    base.CROP = (0, 125, 1280, 720)
    (OUT / "review").mkdir(parents=True)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    for i, row in enumerate(rows, 1):
        source = sources[row["video"]]
        dest = OUT / f"G_{Path(row['video']).stem}_{row['second']:05d}.png"
        row["file"] = dest.name
        row["crop_xywh_native"] = list(base.CROP)
        row["source_sha256"] = source["sha256"]
        row.update(base.decode(Path(source["file"]), row["second"], dest, exe))
        row["decision"] = "pending_visual_review"
        if i % 16 == 0:
            print(f"decoded {i}/{len(rows)}", flush=True)
    report = {"schema": "style-target-gap-scout-v11", "status": "DRAFT_NOT_FOR_TRAINING",
              "source_scan": str(base.SCAN), "source_inventory": str(base.INVENTORY),
              "parent_candidate_manifest": str(ROOT / "data/processed/style-target-20261001-v9-leftcrop-draft/review/candidate-manifest.json"),
              "crop_xywh_native": list(base.CROP),
              "selection": "Unreviewed time gaps: highest 5-second scan sharpness in each 15-second bin; requires visual review",
              "rows": rows, "contact_pages": base.contacts(rows)}
    (OUT / "review/candidate-manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidates={len(rows)} contacts={len(report['contact_pages'])}")


if __name__ == "__main__":
    main()
