"""Verify exported generator weights reproduce a delivered 720p comparison pixel for pixel."""
from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image
import torch

from .cut_wide_model import Generator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--comparison", required=True)
    args = parser.parse_args()
    meta = json.load(open(args.metadata, encoding="utf-8"))
    if meta["schema"] != "cut-generator-fp32-v1" or meta["step"] != 2000:
        raise ValueError("Unexpected generator metadata")
    model = Generator(width=meta["generator_width"], blocks=meta["residual_blocks"]).cuda().eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    with Image.open(args.source) as image:
        original = np.asarray(image.convert("RGB")).copy()
    with Image.open(args.comparison) as image:
        delivered = np.asarray(image.convert("RGB")).copy()
    if original.shape != (720, 1280, 3) or delivered.shape != (720, 2560, 3):
        raise ValueError("Unexpected comparison dimensions")
    value = (torch.from_numpy(original).permute(2, 0, 1).float().cuda() - 128) / 128
    output = model(value.unsqueeze(0))[0]
    rendered = ((output * 128 + 128).round().clamp(0, 255).byte()
                .permute(1, 2, 0).cpu().numpy())
    if not np.array_equal(delivered[:, :1280], original):
        raise ValueError("Delivered left panel differs from source")
    if not np.array_equal(delivered[:, 1280:], rendered):
        raise ValueError("Exported weights differ from delivered right panel")
    print("EXPORTED_WEIGHTS_REPRODUCE_COMPARISON_EXACTLY")


if __name__ == "__main__":
    main()
