"""Probe cached AdaIN decoder on approved v14 game references and fixed unseen scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .adain_game_train import FixedStyle
from .adain_model import load_pretrained
from .data import audit_style, image_tensor
from .runtime import configure_workspace, file_hash, write_json


STYLES = ("N_18-54-23_00115.png", "N_18-59-02_00650.png", "N_latest_011.png")
SCENES = (0, 5)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--style-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    style = {Path(row["path"]).name: row for row in audit_style(args.style_manifest.parent)}
    if not all(name in style for name in STYLES):
        raise ValueError("Missing approved game references")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    if file_hash(index["parent_manifest"]) != index["parent_sha256"]:
        raise ValueError("Fixed test index changed")
    source_paths = index["selections"]["primary_test"]["paths"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 500:
        raise ValueError("Expected audited 500-step AdaIN checkpoint")
    encoder, decoder = load_pretrained(configure_workspace() / "cache/pretrained/adain")
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    encoder, decoder = encoder.cuda().eval(), decoder.cuda().eval()
    args.output.mkdir(parents=True)
    rows = []
    tiles = []
    for name in STYLES:
        style_tensor = ((image_tensor(style[name]["path"], "cuda") * 128 + 128) / 255).unsqueeze(0)
        model = FixedStyle(encoder, decoder, style_tensor).cuda().eval()
        for scene in SCENES:
            source = Path(source_paths[scene])
            value = image_tensor(source, "cuda").unsqueeze(0)
            output = model(value)[0]
            original = ((value[0] * 128 + 128).round().clamp(0, 255).byte()
                        .permute(1, 2, 0).cpu().numpy())
            result = ((output * 128 + 128).round().clamp(0, 255).byte()
                      .permute(1, 2, 0).cpu().numpy())
            image_path = args.output / f"comparison_{name.removesuffix('.png')}_{scene:02d}.png"
            Image.fromarray(np.concatenate((original, result), axis=1)).save(image_path)
            rows.append({"style": name, "style_sha256": style[name]["sha256"],
                         "scene": scene, "source": str(source),
                         "source_sha256": file_hash(source), "comparison": str(image_path),
                         "comparison_sha256": file_hash(image_path)})
            if scene == SCENES[0]:
                tiles.append((name, original, result))
    sheet = Image.new("RGB", (1600, 3 * 480), "#101010")
    draw = ImageDraw.Draw(sheet)
    for i, (label, original, result) in enumerate(tiles):
        for j, pixels in enumerate((original, result)):
            x, y = j * 800, i * 480
            sheet.paste(Image.fromarray(pixels).resize((800, 450)), (x, y + 30))
            draw.text((x + 10, y + 8), f"{'Real test input' if j == 0 else 'AdaIN game reference '+label}",
                      fill="white")
    sheet_path = args.output / "probe_summary.jpg"
    sheet.save(sheet_path, quality=95)
    write_json(args.output / "probe.json", {"quality_gate": "VISUAL_REVIEW_REQUIRED",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
        "style_manifest": str(args.style_manifest), "sheet": str(sheet_path),
        "sheet_sha256": file_hash(sheet_path), "comparisons": rows})
    print(sheet_path)


if __name__ == "__main__":
    main()
