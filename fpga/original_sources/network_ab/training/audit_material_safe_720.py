"""Validate dimensions, clothing/marble structure, and motion of the safe 720p path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .runtime import file_hash, write_json


REGIONS = {"person_center": (465, 350, 675, 720),
           "marble_right": (840, 180, 1260, 645)}  # XYXY at 1280x720


def read_video(path: Path) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(path))
    meta = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, meta


def gradient_correlation(a: np.ndarray, b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.Laplacian(gray_a, cv2.CV_32F).ravel()
    gb = cv2.Laplacian(gray_b, cv2.CV_32F).ravel()
    return float(np.corrcoef(ga, gb)[0, 1])


def flicker(source: list[np.ndarray], styled: list[np.ndarray],
            region: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = region
    # Analyze at 320x180 to keep the full 149-transition audit inexpensive.
    roi = (x1 // 4, y1 // 4, x2 // 4, y2 // 4)
    prior_source = prior_style = None
    source_errors, style_errors = [], []
    yy, xx = np.mgrid[0:180, 0:320].astype(np.float32)
    for a, b in zip(source, styled):
        small_source = cv2.resize(a, (320, 180), interpolation=cv2.INTER_AREA)
        small_style = cv2.resize(b, (320, 180), interpolation=cv2.INTER_AREA)
        if prior_source is not None:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(small_source, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prior_source, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.1, 0)
            mx, my = xx + flow[:, :, 0], yy + flow[:, :, 1]
            old_source = cv2.remap(prior_source, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            old_style = cv2.remap(prior_style, mx, my, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            rx1, ry1, rx2, ry2 = roi
            src_err = np.abs(small_source.astype(np.float32) -
                             old_source.astype(np.float32)).mean(2)[ry1:ry2, rx1:rx2]
            sty_err = np.abs(small_style.astype(np.float32) -
                             old_style.astype(np.float32)).mean(2)[ry1:ry2, rx1:rx2]
            stable = src_err < 20
            if stable.any():
                source_errors.append(float(src_err[stable].mean()))
                style_errors.append(float(sty_err[stable].mean()))
        prior_source, prior_style = small_source, small_style
    return {"transitions": len(source_errors),
            "source_flow_l1": float(np.mean(source_errors)),
            "styled_flow_l1": float(np.mean(style_errors)),
            "ratio": float(np.mean(style_errors) / max(np.mean(source_errors), 1e-6))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--styled", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.styled.parent / "status.json").read_text(encoding="utf-8"))
    if (status["video_sha256"] != file_hash(args.styled) or
            status["source_sha256"] != file_hash(args.source)):
        raise ValueError("Audit input drift")
    source, source_meta = read_video(args.source)
    styled, styled_meta = read_video(args.styled)
    if (len(source), len(styled)) != (150, 150):
        raise ValueError("Missing or duplicated timeline frames")
    if source_meta != styled_meta or source_meta != {
            "fps": 30.0, "width": 1920, "height": 1080}:
        raise ValueError(f"Wrong video metadata {source_meta} {styled_meta}")
    same_adjacent = sum(np.array_equal(a, b) for a, b in zip(styled, styled[1:]))
    structural = {}
    for name, (x1, y1, x2, y2) in REGIONS.items():
        samples = [gradient_correlation(source[i][y1:y2, x1:x2],
                                        styled[i][y1:y2, x1:x2])
                   for i in (0, 30, 60, 90, 120, 149)]
        structural[name] = {"gradient_correlation_min": float(min(samples)),
                            "gradient_correlation_mean": float(np.mean(samples)),
                            "flow": flicker(source, styled, (x1, y1, x2, y2))}
    x1, y1, x2, y2 = REGIONS["marble_right"]
    bright_new = []
    for src, sty in zip(source, styled):
        s = cv2.cvtColor(src[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        t = cv2.cvtColor(sty[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        source_white = (s[:, :, 2] >= 235) & (s[:, :, 1] < 70)
        styled_white = (t[:, :, 2] >= 235) & (t[:, :, 1] < 70)
        bright_new.append(float((styled_white & ~source_white).mean()))
    record = {"schema": "material-safe-720-audit-v1",
              "source": str(args.source), "source_sha256": file_hash(args.source),
              "styled": str(args.styled), "styled_sha256": file_hash(args.styled),
              "source_metadata": source_meta, "styled_metadata": styled_meta,
              "frames": len(styled), "identical_consecutive_frames": int(same_adjacent),
              "regions_1280x720": {k: list(v) for k, v in REGIONS.items()},
              "structural": structural,
              "new_white_fraction_marble_mean": float(np.mean(bright_new)),
              "interpretation": "Appearance-safety audit only; high correlation and low flow error do not prove game-style strength."}
    write_json(args.styled.parent / "quality_audit.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
