"""Motion-align only the background of a people-locked inpaint video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from .postprocess_teacher_video import model_mask, warp_errors
from .probe_temporal_stabilization import stabilize
from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--inpaint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    root = configure_workspace()
    original = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    inpaint = json.loads((args.inpaint_dir / "review.json").read_text(encoding="utf-8"))
    if original["source_frame_indices"] != inpaint["indices"]:
        raise ValueError("Inpaint does not cover the same sequence")
    if original["lora_sha256"] != inpaint["lora_sha256"]:
        raise ValueError("Inpaint uses different LoRA weights")
    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    if file_hash(checkpoint) != inpaint["segmenter_sha256"]:
        raise ValueError("Person segmenter changed")
    model = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "source_raw_locked_stabilized.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             original["fps"], (2304, 456))
    if not writer.isOpened():
        raise RuntimeError("Could not create video")
    previous_source = previous_locked = previous_stable = None
    errors = []
    try:
        for position, index in enumerate(inpaint["indices"]):
            source = np.asarray(Image.open(args.input_dir /
                                           f"source_{index:06d}.png").convert("RGB"))
            locked = np.asarray(Image.open(args.inpaint_dir /
                                           f"locked_{index:06d}.png").convert("RGB"))
            if previous_source is None:
                stable = locked.copy()
            else:
                blended, _ = stabilize(previous_source, source,
                                       previous_stable, locked, args.alpha)
                person = model_mask(model, source)[:, :, None]
                stable = np.clip(blended.astype(np.float32) * (1 - person) +
                                 locked.astype(np.float32) * person,
                                 0, 255).astype(np.uint8)
                errors.append(warp_errors(previous_source, source,
                                          previous_locked, locked,
                                          previous_stable, stable))
            panel = np.zeros((456, 2304, 3), dtype=np.uint8)
            panel[:] = 21
            panel[24:, :768] = source
            panel[24:, 768:1536] = locked
            panel[24:, 1536:] = stable
            for x, label in ((12, "SOURCE"), (780, "PEOPLE-LOCKED RAW"),
                             (1548, "LOCKED + BACKGROUND FLOW")):
                cv2.putText(panel, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if position in (0, len(inpaint["indices"]) // 2,
                            len(inpaint["indices"]) - 2,
                            len(inpaint["indices"]) - 1):
                Image.fromarray(panel).save(
                    args.output_dir / f"review_{position:02d}.jpg", quality=95)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            previous_source, previous_locked, previous_stable = source, locked, stable
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(target))
    encoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = sum(capture.read()[0] for _ in range(encoded))
    capture.release()
    if encoded != len(inpaint["indices"]) or decoded != encoded:
        raise RuntimeError("Output video did not decode fully")
    result = {
        "schema": "people-locked-stabilization-v1",
        "status": "visual_review_required",
        "source_inpaint": str(args.inpaint_dir / "source_raw_people_locked.mp4"),
        "source_inpaint_sha256": inpaint["video_sha256"],
        "lora_sha256": inpaint["lora_sha256"],
        "alpha": args.alpha, "frames": encoded, "fps": original["fps"],
        "flow_aligned_l1": {key: float(np.mean([row[key] for row in errors]))
                            for key in ("source_l1", "raw_l1", "corrected_l1")},
        "video": str(target), "video_sha256": file_hash(target),
        "limitation": "Original person pixels preserve identity; only background is styled. Motion blending may soften foliage and does not make the background diffusion temporally aware.",
    }
    write_json(args.output_dir / "review.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
