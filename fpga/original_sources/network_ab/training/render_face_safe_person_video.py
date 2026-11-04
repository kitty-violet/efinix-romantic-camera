"""Assemble a no-loop video with styled people and source-faithful faces."""
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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path, required=True)
    parser.add_argument("--face-safe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-alpha", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 <= args.background_alpha <= 1:
        raise ValueError("background alpha must be in [0, 1]")
    root = configure_workspace()
    original = json.loads((args.input_dir / "status.json").read_text(encoding="utf-8"))
    locked_state = json.loads((args.locked_dir / "review.json").read_text(encoding="utf-8"))
    face_state = json.loads((args.face_safe_dir / "review.json").read_text(encoding="utf-8"))
    indices = original["source_frame_indices"]
    if indices != locked_state["indices"] or indices != [row["index"] for row in face_state["records"]]:
        raise ValueError("Frame sequences differ")
    checkpoint = root / "cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth"
    if file_hash(checkpoint) != face_state["segmenter_sha256"]:
        raise ValueError("Person segmenter changed")
    segmenter = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21)
    segmenter.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    segmenter.eval()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "source_locked_styled_people.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             original["fps"], (2304, 456))
    if not writer.isOpened():
        raise RuntimeError("Could not create video")
    previous_source = previous_locked = previous_final = None
    flow_metrics = []
    person_change = []
    face_change = []
    try:
        for position, (index, face_record) in enumerate(zip(indices, face_state["records"])):
            source = np.asarray(Image.open(args.input_dir /
                                           f"source_{index:06d}.png").convert("RGB"))
            locked = np.asarray(Image.open(args.locked_dir /
                                           f"locked_{index:06d}.png").convert("RGB"))
            styled = np.asarray(Image.open(args.face_safe_dir /
                                           f"face_safe_{index:06d}.png").convert("RGB"))
            person = model_mask(segmenter, source)
            if previous_source is None:
                final = styled.copy()
            else:
                blended, _ = stabilize(previous_source, source, previous_final,
                                       styled, args.background_alpha)
                final = np.clip(blended.astype(np.float32) * (1 - person[:, :, None]) +
                                styled.astype(np.float32) * person[:, :, None],
                                0, 255).astype(np.uint8)
                flow_metrics.append(warp_errors(previous_source, source,
                                                previous_locked, locked,
                                                previous_final, final))
            Image.fromarray(final).save(args.output_dir / f"final_{index:06d}.png")
            foreground = person > 0.5
            if foreground.any():
                person_change.append(float(np.abs(final.astype(np.float32) -
                                                   locked.astype(np.float32))[foreground].mean()))
            for face in face_record["faces"]:
                x, y, width, height = face["box"]
                face_change.append({"index": index,
                                    "source_mae": float(np.abs(
                                        final[y:y + height, x:x + width].astype(np.float32) -
                                        source[y:y + height, x:x + width].astype(np.float32)).mean())})
            panel = np.zeros((456, 2304, 3), dtype=np.uint8)
            panel[:] = 21
            panel[24:, :768] = source
            panel[24:, 768:1536] = locked
            panel[24:, 1536:] = final
            for x, label in ((12, "SOURCE"), (780, "PERSON UNCHANGED"),
                             (1548, "STYLED PEOPLE + FACE PROTECTION")):
                cv2.putText(panel, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if position in (0, len(indices) // 2, len(indices) - 2, len(indices) - 1):
                Image.fromarray(panel).save(args.output_dir / f"review_{position:02d}.jpg", quality=96)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            previous_source, previous_locked, previous_final = source, locked, final
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(target))
    encoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = sum(capture.read()[0] for _ in range(encoded))
    capture.release()
    if encoded != len(indices) or decoded != encoded:
        raise RuntimeError("Output video did not decode fully")
    result = {
        "schema": "face-safe-styled-people-video-v1", "status": "visual_review_required",
        "frames": encoded, "fps": original["fps"],
        "lora_sha256": original["lora_sha256"],
        "person_diffusion_strength": face_state["strength"],
        "background_alpha": args.background_alpha,
        "mean_person_change_from_locked_mae": float(np.mean(person_change)),
        "face_source_mae": face_change,
        "flow_aligned_l1": {key: float(np.mean([row[key] for row in flow_metrics]))
                            for key in ("source_l1", "raw_l1", "corrected_l1")},
        "video": str(target), "video_sha256": file_hash(target),
        "limitation": "Source head geometry is preserved; game appearance of the face remains partly photographic and framewise person diffusion may flicker.",
    }
    write_json(args.output_dir / "review.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
