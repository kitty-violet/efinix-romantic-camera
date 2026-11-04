"""Render a clearly marked failed-quality CUT diagnostic validation video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cut_wide_model import Generator, ResidualGenerator
from .data import load_content_manifest
from .evaluate import render_video
from .runtime import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest = load_content_manifest(config["content_manifest"])
    videos = [v for v in manifest["videos"] if v["split"] == "val"
              and Path(v["path"]).name == args.source_name]
    if len(videos) != 1 or file_hash(videos[0]["path"]) != videos[0]["sha256"]:
        raise ValueError("Fixed validation source not found or changed")
    model_type = {"resnet": Generator, "residual": ResidualGenerator}[
        config.get("generator_arch", "resnet")]
    model = model_type(width=config["generator_width"],
                       blocks=config["residual_blocks"]).cuda().eval()
    model.load_state_dict(checkpoint["generator"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = render_video(model, videos[0]["path"], args.output,
                          seconds=args.seconds, hlg_exposure_stops=-1)
    result.update(checkpoint=str(args.checkpoint), step=checkpoint["step"],
                  quality_gate="FAIL_DIAGNOSTIC_ONLY")
    write_json(args.output.with_suffix(".json"), result)
    print(json.dumps({k: result[k] for k in ("frames", "fps", "preview", "sha256",
                                                "quality_gate")}, indent=2))


if __name__ == "__main__":
    main()
