"""Search the full target timeline for clear scenes resembling held-out content."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .runtime import configure_workspace, file_hash, write_json


def contact(path: Path, candidates: list[dict]) -> None:
    width, height = 480, 292
    sheet = Image.new("RGB", (width * 4, height * 5), "#171717")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(candidates[:20]):
        x, y = (i % 4) * width, (i // 4) * height
        image = Image.open(row["thumbnail"]).convert("RGB").resize((480, 270))
        sheet.paste(image, (x, y + 22))
        draw.text((x + 8, y + 5),
                  f'{row["video"]} {row["second"]:.0f}s  sim {row["similarity"]:.3f} '
                  f'lap {row["overall_laplacian"]:.0f}', fill="white")
    sheet.save(path, quality=92)


def diverse_top(rows: list[dict], scores: np.ndarray, minimum_laplacian=200) -> list[dict]:
    order = np.argsort(-scores)
    selected = []
    for idx in order:
        item = rows[int(idx)]
        if item["overall_laplacian"] < minimum_laplacian:
            continue
        if any(old["video"] == item["video"] and
               abs(old["second"] - item["second"]) < 20 for old in selected):
            continue
        selected.append({key: item[key] for key in
                         ("video", "second", "thumbnail", "thumbnail_sha256",
                          "overall_laplacian", "median_tile_laplacian",
                          "usable_tile_count")} |
                        {"similarity": float(scores[int(idx)])})
        if len(selected) >= 20:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-sources", action="store_true",
                        help="Include the two added Dying Light videos and the final short recording")
    args = parser.parse_args()
    root = configure_workspace()
    timeline_path = root / "runs/style-target-full-timeline-20261001/timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    rows = list(timeline["rows"])
    if len(rows) != 1067:
        raise ValueError("Unexpected target timeline version")
    timeline_paths = [timeline_path]
    if args.all_sources:
        extra_path = root / "runs/style-target-new-sources-20261001/timeline.json"
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        if len(extra["rows"]) != 451:
            raise ValueError("Unexpected added-video timeline version")
        rows.extend(extra["rows"])
        timeline_paths.append(extra_path)
        latest_path = root / "runs/style-target-new-sources-20261001/latest-video-review/candidate-manifest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if len(latest["rows"]) != 19:
            raise ValueError("Unexpected final-video timeline version")
        rows.extend({
            "video": row["video"], "second": float(row["second"]),
            "thumbnail": str(latest_path.parent / row["file"]),
            "thumbnail_sha256": row["sha256"],
            "overall_laplacian": float(row["laplacian_variance_native"]),
            "median_tile_laplacian": float(row["best_4_tile_laplacian_native"]),
            "usable_tile_count": 4,
        } for row in latest["rows"])
        timeline_paths.append(latest_path)
    for row in rows:
        thumbnail = Path(row["thumbnail"])
        if not thumbnail.is_file() or file_hash(thumbnail) != row["thumbnail_sha256"]:
            raise ValueError(f"Target thumbnail changed: {thumbnail}")
    preflight = root / "runs/diffusion-teacher-preflight"
    adapter = json.loads((preflight / "ip_adapter_plus.json").read_text(encoding="utf-8"))
    sd15 = json.loads((preflight / "sd15.json").read_text(encoding="utf-8"))
    encoder_path = Path(adapter["snapshot"]) / "models/image_encoder"
    processor = CLIPImageProcessor.from_pretrained(
        Path(sd15["snapshot"]) / "feature_extractor", local_files_only=True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        encoder_path, torch_dtype=torch.float16, use_safetensors=True,
        local_files_only=True).to("cuda").eval()
    torch.set_grad_enabled(False)

    def embed(paths: list[Path]) -> np.ndarray:
        vectors = []
        for start in range(0, len(paths), 16):
            images = [Image.open(path).convert("RGB") for path in paths[start:start + 16]]
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(
                "cuda", dtype=torch.float16)
            with torch.inference_mode():
                value = encoder(pixel_values=pixels).image_embeds.float()
                value = torch.nn.functional.normalize(value, dim=1)
            vectors.append(value.cpu().numpy())
            if (start // 16) % 10 == 0:
                print(f"embedded {min(start + 16, len(paths))}/{len(paths)}", flush=True)
        return np.concatenate(vectors, axis=0)

    targets = embed([Path(row["thumbnail"]) for row in rows])
    index = json.loads((root / "runs/content-quality-eval-20261001/index.json").read_text(encoding="utf-8"))
    sources = {
        "street": Path(index["selections"]["primary_test"]["paths"][1]),
        "park": Path(index["selections"]["validation_scene_coverage"]["paths"][0]),
    }
    out = root / ("runs/style-target-all-source-neighbors-20261002" if args.all_sources
                  else "runs/style-target-neighbor-audit-20261002")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "game-target-semantic-neighbor-audit-v1",
        "timeline": str(timeline_path), "timeline_sha256": file_hash(timeline_path),
        "source_manifests": [{"path": str(path), "sha256": file_hash(path)}
                             for path in timeline_paths],
        "target_count": len(rows), "encoder_revision": adapter["revision"],
        "quality_threshold_laplacian": 200, "diversity_gap_seconds": 20,
        "queries": {},
    }
    for name, path in sources.items():
        query = embed([path])[0]
        scores = targets @ query
        candidates = diverse_top(rows, scores)
        contact(out / f"{name}_neighbors.jpg", candidates)
        report["queries"][name] = {
            "source": str(path), "source_sha256": file_hash(path),
            "candidates": candidates,
            "contact_sheet": str(out / f"{name}_neighbors.jpg"),
        }
    write_json(out / "neighbors.json", report)
    print(out / "neighbors.json", flush=True)


if __name__ == "__main__":
    main()
