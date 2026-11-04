"""Rank all approved game scenes, including new videos, against fixed real holdouts.

This is a retrieval audit, not an automatic approval rule: a high CLIP score
does not establish that UI, weapons or subject classes are usable for training.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .retrieve_game_target_neighbors import contact, diverse_top
from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    root = configure_workspace()
    manifest = root / "data/processed/style-target-20261001-v14-expanded-reviewed/dataset-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("status") != "approved" or data.get("image_count") != len(data["records"]):
        raise ValueError("Expected an internally consistent approved target set")
    rows = []
    for record in data["records"]:
        path = manifest.parent / record["file"]
        if not path.is_file() or file_hash(path) != record["sha256"]:
            raise ValueError(f"Modified target image: {path}")
        rows.append({
            "video": record["source_video"],
            "second": float(record["second"]),
            "thumbnail": str(path),
            "thumbnail_sha256": record["sha256"],
            "overall_laplacian": float(record["laplacian_variance_native"]),
            "median_tile_laplacian": 0.0,
            "usable_tile_count": 0,
            "scene": record.get("scene"),
            "file": record["file"],
        })
    adapter = json.loads((root / "runs/diffusion-teacher-preflight/ip_adapter_plus.json").read_text(encoding="utf-8"))
    sd15 = json.loads((root / "runs/diffusion-teacher-preflight/sd15.json").read_text(encoding="utf-8"))
    processor = CLIPImageProcessor.from_pretrained(
        Path(sd15["snapshot"]) / "feature_extractor", local_files_only=True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        Path(adapter["snapshot"]) / "models/image_encoder",
        torch_dtype=torch.float16, use_safetensors=True,
        local_files_only=True).to("cuda").eval()

    def embed(paths: list[Path]) -> np.ndarray:
        vectors = []
        for start in range(0, len(paths), 12):
            batch = paths[start:start + 12]
            images = [Image.open(path).convert("RGB") for path in batch]
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(
                "cuda", dtype=torch.float16)
            with torch.inference_mode():
                value = encoder(pixel_values=pixels).image_embeds.float()
                value = torch.nn.functional.normalize(value, dim=1)
            vectors.append(value.cpu().numpy())
        return np.concatenate(vectors, axis=0)

    target_vectors = embed([Path(row["thumbnail"]) for row in rows])
    index = json.loads((root / "runs/content-quality-eval-20261001/index.json").read_text(encoding="utf-8"))
    sources = {
        "street": Path(index["selections"]["primary_test"]["paths"][1]),
        "park": Path(index["selections"]["validation_scene_coverage"]["paths"][0]),
    }
    out = root / "runs/style-correction-review-20261002/approved-target-neighbors"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "approved-target-semantic-neighbor-audit-v1",
        "manifest": str(manifest), "manifest_sha256": file_hash(manifest),
        "target_count": len(rows), "source_video_counts": dict(Counter(r["video"] for r in rows)),
        "encoder_revision": adapter["revision"], "queries": {},
        "limitation": "CLIP similarity and Laplacian sharpness do not detect UI, foreground weapons or an exact vehicle-class match; manual visual review is required.",
    }
    for name, source in sources.items():
        query = embed([source])[0]
        candidates = diverse_top(rows, target_vectors @ query, minimum_laplacian=0)
        # Keep the extra record fields after ranking for source traceability.
        lookup = {(row["video"], row["second"], row["thumbnail"]): row for row in rows}
        for candidate in candidates:
            original = lookup[(candidate["video"], candidate["second"], candidate["thumbnail"])]
            candidate["scene"] = original["scene"]
            candidate["file"] = original["file"]
        contact(out / f"{name}_neighbors.jpg", candidates)
        report["queries"][name] = {
            "source": str(source), "source_sha256": file_hash(source),
            "candidates": candidates,
            "top_20_source_counts": dict(Counter(row["video"] for row in candidates)),
            "contact_sheet": str(out / f"{name}_neighbors.jpg"),
        }
    write_json(out / "neighbors.json", report)
    print(out / "neighbors.json", flush=True)


if __name__ == "__main__":
    main()
