"""Compare fixed no-vehicle holdout across real LoRA checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    root = configure_workspace()
    run = root / "runs/game-style-lora-simple-20261002-01"
    selection = root / "runs/simple-scene-stage-20261002/target-selection.json"
    data = json.loads(selection.read_text(encoding="utf-8"))
    source = Path(data["fixed_unseen_content"])
    if file_hash(source) != data["fixed_unseen_content_sha256"]:
        raise ValueError("Fixed unseen source changed")
    source_image = Image.open(source).convert("RGB").resize((768, 432))
    target = root / "data/processed/style-target-20261001-v14-expanded-reviewed/C_3_01095.png"
    target_image = Image.open(target).convert("RGB").resize((768, 432))
    paths = {}
    for step in (500, 1000, 1500, 2000):
        tag = ("simple-stage-no-vehicle" if step == 500
               else "simple-stage-checkpoint-compare")
        path = (run / f"probe_weights_{step:06d}_validation_scene_coverage_sample0_{tag}"
                / "teacher_strength_0.70.png")
        state = json.loads((path.parent / "probe_status.json").read_text(encoding="utf-8"))
        if state["source_sha256"] != data["fixed_unseen_content_sha256"]:
            raise ValueError("Checkpoint probe used a different source")
        if path.is_file() is False:
            raise FileNotFoundError(path)
        paths[step] = path
    panels = [("Original held-out park", source_image),
              ("Example target (not an inference input)", target_image)]
    panels.extend((f"LoRA {step} steps", Image.open(path).convert("RGB"))
                  for step, path in paths.items())
    out = root / "runs/simple-scene-stage-20261002/checkpoint-review.jpg"
    sheet = Image.new("RGB", (3 * 768, 2 * 458), "#141414")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(panels):
        x, y = i % 3 * 768, i // 3 * 458
        sheet.paste(image, (x, y + 26))
        draw.text((x + 10, y + 6), label, fill="white")
    sheet.save(out, quality=95)
    write_json(out.parent / "checkpoint-review.json", {
        "schema": "simple-stage-checkpoint-review-v1",
        "status": "visual_review_required",
        "fixed_unseen_source": str(source),
        "fixed_unseen_source_sha256": file_hash(source),
        "illustrative_target": str(target), "illustrative_target_sha256": file_hash(target),
        "image": str(out), "image_sha256": file_hash(out),
        "checkpoint_outputs": {str(step): {"path": str(path), "sha256": file_hash(path)}
                               for step, path in paths.items()},
    })
    print(out, flush=True)


if __name__ == "__main__":
    main()
