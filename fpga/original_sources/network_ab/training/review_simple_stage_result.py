"""Verify starter-stage artifacts and record temporal and visual limitations."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .postprocess_teacher_video import warp_errors
from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    root = configure_workspace()
    stage = root / "runs/simple-scene-stage-20261002"
    run = root / "runs/game-style-lora-simple-20261002-01"
    selection = json.loads((stage / "target-selection.json").read_text(encoding="utf-8"))
    panel = json.loads((stage / "checkpoint-review.json").read_text(encoding="utf-8"))
    video_folder = stage / "temporal-validation"
    video_state = json.loads((video_folder / "status.json").read_text(encoding="utf-8"))
    if video_state["status"] != "rendered_visual_review_required":
        raise ValueError("Video render did not complete")
    if video_state["anchor_sha256"] != selection["fixed_unseen_content_sha256"]:
        raise ValueError("Video and still holdouts differ")
    indices = video_state["source_frame_indices"]
    if len(indices) != 16:
        raise ValueError("Unexpected video sequence length")
    temporal = []
    previous_source = previous_styled = None
    for index in indices:
        source = np.asarray(Image.open(video_folder / f"source_{index:06d}.png").convert("RGB"))
        styled = np.asarray(Image.open(video_folder / f"styled_{index:06d}.png").convert("RGB"))
        if previous_source is not None:
            temporal.append(warp_errors(previous_source, source,
                                        previous_styled, styled,
                                        previous_styled, styled))
        previous_source, previous_styled = source, styled
    video_path = Path(video_state["video"])
    if file_hash(video_path) != video_state["video_sha256"]:
        raise ValueError("Video hash changed")
    capture = cv2.VideoCapture(str(video_path))
    info = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    capture.release()
    if info != [32, 1536, 456]:
        raise ValueError(f"Unexpected video dimensions or length: {info}")
    losses = np.asarray(json.loads((run / "losses.json").read_text(encoding="utf-8"))["losses"])
    if len(losses) != 2000 or not np.isfinite(losses).all():
        raise ValueError("Training loss history incomplete")
    write_json(stage / "quality-review.json", {
        "schema": "simple-stage-quality-review-v1",
        "status": "trained_but_not_accepted",
        "reason": "The no-vehicle trial removes the motorcycle-shape confound. The 500-step result adds some rendered foliage and pavement appearance, but the Dying Light target look remains weak and people shift or soften. Higher checkpoints do not consistently improve visual quality. The rendered sequence still needs temporal correction.",
        "best_visual_checkpoint_for_this_probe": 500,
        "best_weight": str(run / "weights_000500/pytorch_lora_weights.safetensors"),
        "best_weight_sha256": file_hash(run / "weights_000500/pytorch_lora_weights.safetensors"),
        "final_trained_checkpoint": 2000,
        "final_weight": str(run / "weights_002000/pytorch_lora_weights.safetensors"),
        "final_weight_sha256": file_hash(run / "weights_002000/pytorch_lora_weights.safetensors"),
        "training_target_selection": str(stage / "target-selection.json"),
        "target_count": selection["image_count"],
        "fixed_unseen_content": selection["fixed_unseen_content"],
        "checkpoint_panel": panel["image"],
        "checkpoint_panel_sha256": panel["image_sha256"],
        "loss_means_per_500_steps": [float(losses[i:i + 500].mean())
                                     for i in range(0, 2000, 500)],
        "temporal_video": str(video_path),
        "temporal_video_sha256": video_state["video_sha256"],
        "temporal_video_info": {"encoded_frames": info[0], "unique_frames": len(indices),
                                "width": info[1], "height": info[2], "fps": 10},
        "flow_aligned_l1": {
            "source": float(np.mean([row["source_l1"] for row in temporal])),
            "teacher": float(np.mean([row["raw_l1"] for row in temporal])),
        },
        "decision": "Keep all genuine checkpoints and comparisons. Do not promote these as accepted game-style weights. The next experiment should change the training objective/conditioning, not merely extend the same LoRA steps.",
    })
    print(stage / "quality-review.json", flush=True)


if __name__ == "__main__":
    main()
