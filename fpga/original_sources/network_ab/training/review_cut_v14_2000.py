"""Render fixed, source-disjoint 720p evidence for the 2000-step v14 CUT run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .cut_wide_model import Generator
from .evaluate import render_video
from .runtime import file_hash, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--game-reference", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if state["step"] != 2000 or config["expected_style_count"] != 83:
        raise ValueError("Expected the completed v14 2000-step checkpoint")
    if file_hash(config["style_manifest"]) != state["binding"]["style_sha256"]:
        raise ValueError("Target manifest changed since training")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    if file_hash(index["parent_manifest"]) != index["parent_sha256"]:
        raise ValueError("Content evaluation index changed")
    paths = index["selections"]["primary_test"]["paths"]
    if len(paths) < 16:
        raise ValueError("Insufficient fixed test scenes")
    model = Generator(width=config["generator_width"],
                      blocks=config["residual_blocks"]).cuda().eval()
    model.load_state_dict(state["generator"])
    args.output.mkdir(parents=True)
    records = []
    choices = (0, 5, 11, 13)
    tiles = []
    for number, index_number in enumerate(choices):
        path = Path(paths[index_number])
        with Image.open(path) as image:
            original = np.asarray(image.convert("RGB")).copy()
        if original.shape != (720, 1280, 3):
            raise ValueError(f"Unexpected test size: {path}")
        value = (torch.from_numpy(original).permute(2, 0, 1).float().cuda() - 128) / 128
        output = model(value.unsqueeze(0))[0]
        result = ((output * 128 + 128).round().clamp(0, 255).byte()
                  .permute(1, 2, 0).cpu().numpy())
        comparison = args.output / f"comparison_{number:02d}.png"
        Image.fromarray(np.concatenate((original, result), axis=1)).save(comparison)
        tiles.append((original, result))
        records.append({"source": str(path), "source_sha256": file_hash(path),
                        "comparison": str(comparison), "comparison_sha256": file_hash(comparison),
                        "input_rgb_mean": original.mean((0, 1)).round(2).tolist(),
                        "output_rgb_mean": result.mean((0, 1)).round(2).tolist()})
    # Each row is one source: left is real input, right is the generated result.
    sheet = Image.new("RGB", (1600, len(tiles) * 480), "#101010")
    draw = ImageDraw.Draw(sheet)
    for row, (original, result) in enumerate(tiles):
        for col, pixels in enumerate((original, result)):
            x, y = col * 800, row * 480
            sheet.paste(Image.fromarray(pixels).resize((800, 450)), (x, y + 30))
            draw.text((x + 10, y + 8),
                      f"{'Real test input' if col == 0 else '2000-step output'} / scene {row + 1}",
                      fill="white")
    summary = args.output / "test_summary.jpg"
    sheet.save(summary, quality=95)
    current_validation = args.checkpoint.parent / "validation_002000/comparison_02.png"
    prior = np.asarray(Image.open(args.prior_comparison).convert("RGB"))
    current = np.asarray(Image.open(current_validation).convert("RGB"))
    if prior.shape != current.shape or prior.shape != (720, 2560, 3):
        raise ValueError("Comparison images do not have matching full-frame dimensions")
    if not np.array_equal(prior[:, :1280], current[:, :1280]):
        raise ValueError("Prior and current validations use different scene content")
    reference = args.game_reference.resolve()
    style_manifest = json.loads(Path(config["style_manifest"]).read_text(encoding="utf-8"))
    valid_references = {record["file"]: record["sha256"] for record in style_manifest["records"]}
    if reference.parent != Path(config["style_manifest"]).parent.resolve() or \
            file_hash(reference) != valid_references.get(reference.name):
        raise ValueError("Game reference is not in the approved v14 target set")
    with Image.open(reference) as image:
        game = image.convert("RGB")
    panel = Image.new("RGB", (1600, 960), "#101010")
    panel_draw = ImageDraw.Draw(panel)
    panel_tiles = (("Real input, same validation scene", Image.fromarray(current[:, :1280])),
                   ("Game style reference, different scene", game),
                   ("Prior v7 target, 500 steps", Image.fromarray(prior[:, 1280:])),
                   ("New v14 target, 2000 steps", Image.fromarray(current[:, 1280:])))
    for i, (label, image) in enumerate(panel_tiles):
        x, y = (i % 2) * 800, (i // 2) * 480
        panel.paste(image.resize((800, 450)), (x, y + 30))
        panel_draw.text((x + 10, y + 8), label, fill="white")
    panel_path = args.output / "prior_vs_2000.jpg"
    panel.save(panel_path, quality=95)
    content = json.loads(Path(config["content_manifest"]).read_text(encoding="utf-8"))
    test_videos = [row for row in content["videos"] if row["split"] == "test"
                   and Path(row["path"]).name == "VID_20260925_070032.mp4"]
    if len(test_videos) != 1 or file_hash(test_videos[0]["path"]) != test_videos[0]["sha256"]:
        raise ValueError("Fixed test video missing or modified")
    video = args.output / "test_day_4s_side_by_side.mp4"
    video_result = render_video(model, test_videos[0]["path"], video,
                                seconds=4, hlg_exposure_stops=-1)
    write_json(args.output / "review.json", {
        "step": 2000, "quality_gate": "VISUAL_REVIEW_REQUIRED",
        "note": "Unpaired game targets have different scene content; this is a fixed unseen test set.",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
        "style_manifest": config["style_manifest"], "test_images": records,
        "summary": str(summary), "summary_sha256": file_hash(summary),
        "prior_vs_2000": str(panel_path), "prior_vs_2000_sha256": file_hash(panel_path),
        "prior_comparison_sha256": file_hash(args.prior_comparison),
        "game_reference": str(reference), "game_reference_sha256": file_hash(reference),
        "video": video_result,
    })
    print(json.dumps({"summary": str(summary), "panel": str(panel_path), "video": str(video),
                      "frames": video_result["frames"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
