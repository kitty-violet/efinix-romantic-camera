"""Inventory uploaded sources or prepare 720p masters from a reviewed source plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import IMAGE_SUFFIXES, VIDEO_SUFFIXES, audit_incoming, load_content_manifest
from .runtime import canonical_hash, configure_workspace, file_hash, write_json
from .media import decode_frames, probe_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Reviewed source-level split plan; omitted means inventory only")
    parser.add_argument("--name", default="content-v1")
    args = parser.parse_args()
    root = configure_workspace()
    if not args.plan:
        inventory = {kind: audit_incoming(root / "data/incoming" / kind) for kind in ("content", "holdout")}
        write_json(root / "data/source-inventory.json", {"status": "AWAITING_UPLOAD" if not inventory["content"] else "REVIEW_REQUIRED",
                    "files": inventory, "note": "Assign split by source/recording group; never by individual adjacent frames"})
        print(json.dumps({k: len(v) for k, v in inventory.items()}, ensure_ascii=False))
        return
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "model-training-source-plan-v1" or plan.get("status") != "reviewed":
        raise ValueError("Review inventory, source grouping and scenes before preprocessing")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("Dataset name must be one directory component")
    output = root / "data/processed" / args.name
    output.mkdir(parents=True, exist_ok=False)
    groups, hashes = {}, {}
    for source in plan["sources"]:
        if source["split"] not in ("train", "val", "test") or not source["group"]:
            raise ValueError("Every source needs a nonempty recording group and explicit split")
        path = Path(source["path"]).resolve()
        if (root / "data/incoming").resolve() not in path.parents:
            raise ValueError("Source must reside in the authorized incoming directory")
        if (root / "data/incoming/holdout").resolve() in path.parents and source["split"] == "train":
            raise ValueError("Holdout uploads must never enter training")
        if file_hash(path) != source["sha256"]:
            raise ValueError("Source changed since inventory")
        for value, registry in ((source["group"], groups), (source["sha256"], hashes)):
            if value in registry and registry[value] != source["split"]:
                raise ValueError("Source/recording group crosses splits")
            registry[value] = source["split"]
    records, videos, rejected = [], [], []
    exact = {}
    # Conservative near-duplicate grouping across every split, reviewed thresholds.
    signatures = []
    thresholds = plan["filter_thresholds"]
    def accept(image, source, timestamp, index):
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_wh = list(image.size)
        if abs(image.width / image.height - 16 / 9) > 0.02:
            raise ValueError("Non-16:9 source requires a separately reviewed orientation/crop plan")
        # Crop-to-fill preserves geometry; never stretch to 16:9 or pad black borders.
        image = ImageOps.fit(image, (1280, 720), method=Image.Resampling.BICUBIC)
        array = np.asarray(image)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark = float((gray <= 3).mean())
        bright = float((gray >= 252).mean())
        thumb = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        bits = (thumb[:, 1:] > thumb[:, :-1]).ravel()
        phash = sum(int(bit) << i for i, bit in enumerate(bits))
        mean = array.mean((0, 1)).tolist()
        reason = None
        if blur < thresholds["min_laplacian_variance"]:
            reason = "blur"
        elif max(dark, bright) > thresholds["max_clipped_fraction"]:
            reason = "exposure"
        elif any((phash^old[0]).bit_count() <= thresholds["max_dhash_distance"]
                 and max(abs(a-b) for a, b in zip(mean, old[1])) <= thresholds["max_duplicate_rgb_mean_delta"] for old in signatures):
            reason = "near_duplicate"
        if reason:
            rejected.append({"source": source["path"], "timestamp": timestamp, "reason": reason,
                             "laplacian_variance": blur, "dark_fraction": dark, "bright_fraction": bright})
            return
        destination = output / source["split"] / f"{source['sha256'][:16]}_{index:08d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("Duplicate source or overlapping intervals create duplicate frame names")
        image.save(destination)
        digest = file_hash(destination)
        if digest in exact:
            # Keep recorded evidence; do not silently include a duplicate.
            rejected.append({"path": str(destination), "reason": "exact_duplicate", "same_as": exact[digest]})
            return
        exact[digest] = str(destination)
        signatures.append((phash, mean))
        records.append({"path": str(destination), "sha256": digest, "source_sha256": source["sha256"],
                        "group": source["group"], "split": source["split"], "role": "content",
                        "height": 720, "width": 1280, "source": source["path"], "timestamp": timestamp,
                        "frame_index": index, "original_wh": original_wh,
                        "transform": "EXIF/FFmpeg autorotate; video RGB policy in source_decode; center 16:9 crop; bicubic 1280x720",
                        "source_decode": source.get("decode"),
                        "dhash": f"{phash:016x}", "laplacian_variance": blur,
                        "dark_fraction": dark, "bright_fraction": bright})
    # Process holdouts first so near-duplicate train frames cannot steal a test sample.
    order = {"test": 0, "val": 1, "train": 2}
    for source in sorted(plan["sources"], key=lambda s: (order[s["split"]], s["path"])):
        path = Path(source["path"])
        if path.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(path) as image:
                accept(image, source, None, 0)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            # Phone clips can have variable frame rate. Sample decoded PTS, not
            # index/average-fps, so review timestamps still identify the frame.
            source["decode"] = probe_video(path)
            reader = decode_frames(path)
            try:
                fps = source["decode"]["fps"]
                if not 0 < source["sample_fps"] <= 2 or (fps is not None and source["sample_fps"] > fps):
                    raise ValueError("Choose a reviewed sampling rate up to 2fps")
                next_sample = source.get("start_seconds", 0.0)
                end = source.get("end_seconds", float("inf"))
                for index, timestamp, frame in reader:
                    if timestamp > end:
                        break
                    if timestamp + 1e-6 >= next_sample:
                        if not any(a <= timestamp <= b for a, b in source.get("exclude_intervals", [])):
                            accept(Image.fromarray(frame), source, timestamp, index)
                        next_sample += 1/source["sample_fps"]
            finally:
                reader.close()
            if source["split"] in ("val", "test"):
                videos.append({"path": str(path), "sha256": source["sha256"], "group": source["group"],
                               "split": source["split"]})
        else:
            raise ValueError(f"Unsupported source: {path}")
    manifest = {"schema": "model-training-data-v1", "status": "draft_requires_visual_review",
                "plan_sha256": file_hash(args.plan), "images": records, "videos": videos,
                "filters": thresholds, "rejected": rejected,
                "next": "Review contact sheets and scene coverage; then explicitly approve manifest"}
    write_json(output / "manifest.draft.json", manifest)
    for split in ("train", "val", "test"):
        selected = [r for r in records if r["split"] == split]
        review = output / "review" / split
        review.mkdir(parents=True, exist_ok=True)
        for offset in range(0, len(selected), 20):
            sheet = Image.new("RGB", (1280, 5*202), "#202020")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(selected[offset:offset+20]):
                x, y = index % 4*320, index // 4*202
                with Image.open(row["path"]) as image:
                    sheet.paste(image.resize((320, 180)), (x, y))
                timestamp = row["timestamp"]
                draw.text((x+4, y+182), f"{offset+index:03d} {Path(row['source']).stem[-6:]} {timestamp:.3f}s" if timestamp is not None else Path(row["path"]).stem, fill="white")
            sheet.save(review / f"contact_{offset//20:04d}.jpg", quality=90)
    print(json.dumps({"draft": str(output / "manifest.draft.json"), "accepted": len(records), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
