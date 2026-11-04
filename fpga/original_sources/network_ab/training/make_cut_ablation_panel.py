"""Create a clearly labelled visual record of controlled CUT experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace()
    runs = root / "runs"
    entries = [
        ("Input v4 (-1 EV HLG)", None),
        ("Game reference (different scene)", root / "data/processed/style-target-20261001-v6-daylight-draft/W_3_00225.png"),
        ("CUT + all target frames", runs / "cut-wide-20261001-04-exposure/validation_000500/comparison_02.png"),
        ("CUT + daylight targets", runs / "cut-wide-20261001-05-daylight/validation_000500/comparison_02.png"),
        ("Residual generator", runs / "cut-wide-20261001-06-residual/validation_000500/comparison_02.png"),
        ("VGG-content GAN", runs / "cut-wide-20261001-07-vgg/validation_000500/comparison_02.png"),
    ]
    source = np.asarray(Image.open(entries[2][1]).convert("RGB"))[:, :1280]
    canvas = Image.new("RGB", (3 * 480, 2 * 295), "#111111")
    draw = ImageDraw.Draw(canvas)
    files = []
    for i, (label, path) in enumerate(entries):
        if path is None:
            pixels = source
        elif i == 1:
            pixels = np.asarray(Image.open(path).convert("RGB"))
        else:
            image = np.asarray(Image.open(path).convert("RGB"))
            if not np.array_equal(image[:, :1280], source):
                raise ValueError(f"Ablation input mismatch: {path}")
            pixels = image[:, 1280:]
        x, y = i % 3 * 480, i // 3 * 295
        canvas.paste(Image.fromarray(pixels).resize((480, 270)), (x, y + 25))
        draw.text((x + 8, y + 6), label, fill="white")
        if path is not None:
            files.append({"path": str(path), "sha256": file_hash(path)})
    folder = runs / "cut-wide-diagnostic-20261001"
    folder.mkdir(exist_ok=True)
    output = folder / "ablation_panel.jpg"
    canvas.save(output, quality=94)
    write_json(folder / "ablation_panel.json", {"output": str(output),
        "sha256": file_hash(output), "input": "v4 exposure-corrected plaza validation frame",
        "target_unpaired": True, "quality_gate": "FAIL", "sources": files})
    print(output)


if __name__ == "__main__":
    main()
