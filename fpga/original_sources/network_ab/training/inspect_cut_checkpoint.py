"""Render a CUT checkpoint on source-disjoint manifest images for diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .cut_wide_model import Generator, ResidualGenerator
from .data import load_content_manifest
from .runtime import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="Exact source video filename; repeat to compare scenes")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest = load_content_manifest(config["content_manifest"])
    arch = config.get("generator_arch", "resnet")
    model_type = {"resnet": Generator, "residual": ResidualGenerator}[arch]
    model = model_type(width=config["generator_width"],
                       blocks=config["residual_blocks"]).cuda().eval()
    model.load_state_dict(checkpoint["generator"])
    args.output.mkdir(parents=True)
    results = []
    for index, name in enumerate(args.source):
        rows = [row for row in manifest["images"] if row["split"] == args.split
                and Path(row["source"]).name == name]
        if not rows:
            raise ValueError(f"No {args.split} images for {name}")
        row = rows[len(rows) // 2]
        with Image.open(row["path"]) as image:
            before = np.asarray(image.convert("RGB")).copy()
        input_image = (torch.from_numpy(before).permute(2, 0, 1).float()
                       .unsqueeze(0).cuda() - 128) / 128
        with torch.inference_mode():
            output = model(input_image)
        after = ((output[0].clamp(-1, 127 / 128) * 128 + 128)
                 .round().byte().permute(1, 2, 0).cpu().numpy())
        path = args.output / f"comparison_{index:02d}_{Path(name).stem}.png"
        Image.fromarray(np.concatenate([before, after], axis=1)).save(path)
        results.append({"source": name, "input": row["path"], "comparison": str(path),
                        "comparison_sha256": file_hash(path), "split": args.split,
                        "mean_absolute_rgb_change": float(np.abs(after.astype(np.int16)
                            - before.astype(np.int16)).mean())})
    write_json(args.output / "review.json", {"checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint), "step": checkpoint["step"],
        "config": config, "results": results})
    print(json.dumps({"step": checkpoint["step"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
