"""Build the fixed-scene v6/v7 comparison and record the visual quality gate."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace()
    runs = root / "runs"
    v6 = runs / "cut-wide-20261001-05-daylight/validation_000500/comparison_02.png"
    v7_run = runs / "cut-wide-20261001-08-wide-reviewed"
    v7 = v7_run / "validation_000500/comparison_02.png"
    a = np.asarray(Image.open(v6).convert("RGB"))
    b = np.asarray(Image.open(v7).convert("RGB"))
    if a.shape != (720, 2560, 3) or not np.array_equal(a[:, :1280], b[:, :1280]):
        raise ValueError("Controlled validation scene mismatch")
    game = root / "data/processed/style-target-20261001-v7-wide-reviewed/W_N_3_00560.png"
    tiles = [
        ("Real input, exposure corrected", Image.fromarray(a[:, :1280])),
        ("Game reference, different scene", Image.open(game).convert("RGB")),
        ("Prior daylight target, 500 steps", Image.fromarray(a[:, 1280:])),
        ("Expanded panorama target, 500 steps", Image.fromarray(b[:, 1280:])),
    ]
    canvas = Image.new("RGB", (1600, 960), "#101010")
    draw = ImageDraw.Draw(canvas)
    for i, (label, photo) in enumerate(tiles):
        x, y = (i % 2) * 800, (i // 2) * 480
        canvas.paste(photo.resize((800, 450)), (x, y + 30))
        draw.text((x + 12, y + 9), label, fill="white")
    out = runs / "cut-wide-diagnostic-20261001/v7_fixed_scene_review.jpg"
    canvas.save(out, quality=94)
    write_json(out.with_suffix(".json"), {"status": "FAILED_VISUAL_REVIEW",
        "reason": "Purple/cyan sky, colored pavement, coarse false tree texture; panorama expansion did not deliver game-model material",
        "unpaired_game_reference": True, "same_validation_source": True,
        "sources_sha256": {str(p): file_hash(p) for p in (v6, v7, game)},
        "panel_sha256": file_hash(out)})
    status_path = v7_run / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "STOP_AFTER_REACHED" or status["step"] != 500:
        raise ValueError("Unexpected v7 run state")
    checkpoint = v7_run / "checkpoint_000500.pt"
    status.update(quality_gate="FAILED_VISUAL_REVIEW",
        reason="Panorama expansion improved coverage but failed style quality: false grain, colored pavement and sky",
        review_checkpoint_sha256=file_hash(checkpoint),
        review_comparison_sha256=file_hash(v7), review_panel=str(out),
        delivery_policy="Diagnostic evidence only; not a qualified style model")
    write_json(status_path, status)
    print(out)


if __name__ == "__main__":
    main()
