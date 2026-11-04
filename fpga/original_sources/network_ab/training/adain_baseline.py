"""Real 720p visual benchmark using audited game crops and released AdaIN weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .adain_model import load_pretrained, style_transfer
from .data import load_content_manifest
from .runtime import configure_workspace, file_hash, write_json


def image(path):
    with Image.open(path) as opened:
        array = np.array(opened.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda(), array


def main():
    root = configure_workspace()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = root / "runs/adain-game-baseline-20261001-02"
    run.mkdir(parents=True, exist_ok=False)
    enc, dec = load_pretrained(root / "cache/pretrained/adain")
    enc, dec = enc.cuda(), dec.cuda()
    content_manifest = load_content_manifest(root / "data/processed/content-20260930-v2/manifest.approved.json")
    validation = [r for r in content_manifest["images"] if r["split"] == "val"]
    content, source = image(validation[8]["path"])
    if content.shape != (1, 3, 720, 1280):
        raise ValueError("Expected fixed 720p validation frame")
    manifest = json.loads((root / "data/processed/style-target-20261001-v1/dataset-manifest.json").read_text(encoding="utf-8"))
    references = ("1-00020-C", "2-00540-F", "3-02760-D")
    rows = []
    with torch.no_grad():
        for item in references:
            record = next((r for r in manifest["records"] if r["file"].endswith(item + ".png")), None)
            if record is None:
                raise ValueError(f"Missing audited game reference: {item}")
            path = root / "data/processed/style-target-20261001-v1" / record["file"]
            if file_hash(path) != record["sha256"]:
                raise ValueError("Reference hash changed")
            style, _ = image(path)
            for alpha in (0.75, 1.0):
                output = style_transfer(enc, dec, content, style, alpha)[0]
                pixels = output.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                picture = run / f"comparison_{item}_{alpha:.2f}.jpg"
                Image.fromarray(np.concatenate((source, pixels), axis=1)).save(picture, quality=94)
                rows.append({"reference": item, "reference_sha256": record["sha256"],
                             "alpha": alpha, "comparison": str(picture), "sha256": file_hash(picture),
                             "input_rgb_mean": source.mean(axis=(0, 1)).tolist(),
                             "output_rgb_mean": pixels.mean(axis=(0, 1)).tolist()})
    write_json(run / "results.json", {"model": "public_pretrained_AdaIN_unmodified",
                                     "trained_on_user_game": False,
                                     "content": validation[8], "results": rows,
                                     "purpose": "Choose visual direction before custom training"})
    print(str(run / "results.json"))


if __name__ == "__main__":
    main()
