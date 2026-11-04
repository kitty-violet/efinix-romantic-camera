"""Rebuild already reviewed frame selections through the corrected RGB pipeline.

Preserve source grouping, frame order and exclusions. Never silently approve a
new pixel dataset: write a draft plus contact sheets for a new visual review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .data import load_content_manifest
from .media import decode_frames, decoder_identity, probe_video
from .runtime import configure_workspace, file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hlg-exposure-stops", type=float, default=0.0)
    args = parser.parse_args()
    root = configure_workspace()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("Dataset name must be one directory component")
    if not -3.0 <= args.hlg_exposure_stops <= 0.0:
        raise ValueError("HLG exposure must be between -3 and 0 stops")
    parent = load_content_manifest(args.manifest)
    output = root / "data/processed" / args.name
    output.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(parent)
    manifest.update(status="draft_requires_visual_review", parent_manifest=str(args.manifest),
        parent_manifest_sha256=file_hash(args.manifest), decoder=decoder_identity(),
        correction=f"Same reviewed source/frame selections; HLG exposure {args.hlg_exposure_stops:g} stops before Mobius tone mapping",
        selection_policy="Retain parent selections; new color/exposure statistics require review")
    manifest["hlg_exposure_stops"] = args.hlg_exposure_stops
    sources = defaultdict(list)
    for position, row in enumerate(parent["images"]):
        sources[row["source"]].append((position, row))
    audit = []
    for source, selected in sources.items():
        path = Path(source).resolve()
        if (root / "data/incoming").resolve() not in path.parents:
            raise ValueError("Source outside authorized upload directory")
        if any(file_hash(path) != row["source_sha256"] for _, row in selected[:1]):
            raise ValueError("Original video hash changed")
        metadata = probe_video(path)
        indices = {int(Path(row["path"]).stem.rsplit("_", 1)[1]): (position, row)
                   for position, row in selected}
        if len(indices) != len(selected):
            raise ValueError("Parent contains duplicate source frame selection")
        source_audit = {"source": source, "decode": metadata, "frames": []}
        for index, pts, array in decode_frames(path, indices=sorted(indices),
                                               hlg_exposure_stops=args.hlg_exposure_stops):
            position, old = indices[index]
            # showinfo's decimal PTS is rounded; tolerate 0.1ms, not one frame.
            if abs(pts - old["timestamp"]) > 0.0001:
                raise ValueError(f"Frame/PTS drift: {path.name} frame {index}: {pts} != {old['timestamp']}")
            image = Image.fromarray(array)
            if abs(image.width / image.height - 16 / 9) > .02:
                raise ValueError("Unexpected geometry change")
            image = ImageOps.fit(image, (1280, 720), method=Image.Resampling.BICUBIC)
            pixels = np.asarray(image)
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            thumb = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
            bits = (thumb[:, 1:] > thumb[:, :-1]).ravel()
            phash = sum(int(bit) << i for i, bit in enumerate(bits))
            destination = output / old["split"] / Path(old["path"]).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
            row = manifest["images"][position]
            row.update(path=str(destination), sha256=file_hash(destination),
                frame_index=index, timestamp=pts, source_pts_seconds=pts+metadata["timestamp_origin_seconds"],
                timestamp_basis="decoded PTS relative to first displayed frame", parent_image_sha256=old["sha256"],
                transform=f"FFmpeg autorotate; HLG->sRGB Mobius {args.hlg_exposure_stops:g} EV; center 16:9 crop; bicubic 1280x720",
                source_decode=metadata, dhash=f"{phash:016x}",
                laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                dark_fraction=float((gray <= 3).mean()), bright_fraction=float((gray >= 252).mean()))
            source_audit["frames"].append({"frame_index": index, "pts": pts,
                "pts_difference": pts-old["timestamp"],
                "bottom_row_zero_fraction": float((pixels[-1] == 0).all(-1).mean()),
                "top_row_zero_fraction": float((pixels[0] == 0).all(-1).mean()),
                "dark_fraction": row["dark_fraction"], "bright_fraction": row["bright_fraction"],
                "laplacian_variance": row["laplacian_variance"]})
        audit.append(source_audit)
        print(json.dumps({"source": path.name, "corrected_frames": len(indices)}), flush=True)
    for video in manifest["videos"]:
        video["decode"] = probe_video(video["path"])
    for split in ("train", "val", "test"):
        rows = [row for row in manifest["images"] if row["split"] == split]
        directory = output / "review" / split
        directory.mkdir(parents=True)
        for offset in range(0, len(rows), 40):
            sheet = Image.new("RGB", (1600, 8*202), "#161616")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(rows[offset:offset+40]):
                x, y = index % 5*320, index // 5*202
                with Image.open(row["path"]) as image:
                    sheet.paste(image.resize((320, 180)), (x, y))
                draw.text((x+3, y+182), f"{offset+index:03d} {Path(row['source']).stem[-6:]} {row['timestamp']:.3f}s", fill="white")
            sheet.save(directory / f"contact_{offset//40:03d}.jpg", quality=94)
    write_json(output / "manifest.draft.json", manifest)
    write_json(output / "decode-audit.json", {"sources": audit, "images": len(manifest["images"]),
        "all_source_frame_selections_preserved": True, "decoder": decoder_identity()})
    print(str(output / "manifest.draft.json"), flush=True)


if __name__ == "__main__":
    main()
