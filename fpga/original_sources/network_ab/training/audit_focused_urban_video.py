"""Measure visible wall-region temporal changes in the focused teacher video."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .runtime import file_hash, write_json


ROI = (520, 90, 735, 320)  # Right wall, away from the moving foreground person.


def highpass(gray: np.ndarray) -> np.ndarray:
    low = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    return gray.astype(np.float32) - low


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (432, 1536):
            raise ValueError("Wrong diagnostic video size")
        frames.append(frame)
    capture.release()
    if len(frames) != 50:
        raise ValueError(f"Expected 50 frames, got {len(frames)}")
    x1, y1, x2, y2 = ROI
    unaligned = {"source_mae": [], "style_mae": [],
                 "source_highpass_mae": [], "style_highpass_mae": []}
    for previous, current in zip(frames, frames[1:]):
        for name, shift in (("source", 0), ("style", 768)):
            a = cv2.cvtColor(previous[:, shift:shift + 768], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(current[:, shift:shift + 768], cv2.COLOR_BGR2GRAY)
            aa = a[y1:y2, x1:x2].astype(np.float32)
            bb = b[y1:y2, x1:x2].astype(np.float32)
            unaligned[f"{name}_mae"].append(float(np.mean(np.abs(aa - bb))))
            unaligned[f"{name}_highpass_mae"].append(float(np.mean(np.abs(
                highpass(a)[y1:y2, x1:x2] - highpass(b)[y1:y2, x1:x2]))))
    board = Image.new("RGB", (5 * 768, 2 * 432 + 25), "#151515")
    draw = ImageDraw.Draw(board)
    for column, index in enumerate((0, 10, 20, 30, 40)):
        frame = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
        board.paste(Image.fromarray(frame[:, :768]), (column * 768, 25))
        board.paste(Image.fromarray(frame[:, 768:]), (column * 768, 25 + 432))
        draw.text((column * 768 + 8, 5), f"{index / 10:.1f}s source (top) / teacher (bottom)",
                  fill="white")
    contact = args.video.parent / "temporal_contact_0to4s.jpg"
    board.save(contact, quality=90)
    summary = {key: {"mean": float(statistics.mean(values)),
                     "median": float(statistics.median(values))}
               for key, values in unaligned.items()}
    output = {"schema": "focused-urban-wall-flicker-audit-v1",
              "video": str(args.video), "video_sha256": file_hash(args.video),
              "roi_xyxy_768px": list(ROI), "frames": len(frames),
              "method": "Consecutive 10 fps frame absolute difference in the right-wall ROI; no motion compensation, so camera motion contributes to both figures. High-pass difference emphasizes texture instability.",
              "summary": summary, "temporal_contact": str(contact),
              "temporal_contact_sha256": file_hash(contact)}
    write_json(args.video.parent / "flicker_audit.json", output)
    print(summary)


if __name__ == "__main__":
    main()
