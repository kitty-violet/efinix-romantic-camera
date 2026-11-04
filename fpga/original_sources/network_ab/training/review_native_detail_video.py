"""Compare frame-to-frame grass stability and native detail against a raw teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .media import decode_frames
from .postprocess_teacher_video import warp_errors
from .runtime import write_json


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def video_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot decode output frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def focus_energy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--new-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.new_video))
    if not capture.isOpened():
        raise RuntimeError("Cannot open new video")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(int(path.stem.split("_")[-1]) for path in
                     args.teacher_dir.glob("source_*.png")
                     if int(path.stem.split("_")[-1]) < frame_count and
                     (args.teacher_dir /
                      f"styled_{int(path.stem.split('_')[-1]):06d}.png").is_file())
    if len(indices) < 2 or indices[0] != 0 or 45 not in indices:
        raise ValueError("Need at least two matching teacher frames, including 0 and 45")
    crop = np.s_[170:400, 500:768]
    all_scores: list[dict] = []
    grass_scores: list[dict] = []
    previous = None
    for index in indices:
        source = read_rgb(args.teacher_dir / f"source_{index:06d}.png")
        teacher = read_rgb(args.teacher_dir / f"styled_{index:06d}.png")
        native_new = video_frame(capture, index)
        new = cv2.resize(native_new, (768, 432), interpolation=cv2.INTER_AREA)
        if previous is not None:
            prev_source, prev_teacher, prev_new = previous
            all_scores.append(warp_errors(prev_source, source, prev_teacher,
                                          teacher, prev_new, new))
            grass_scores.append(warp_errors(prev_source[crop], source[crop],
                                            prev_teacher[crop], teacher[crop],
                                            prev_new[crop], new[crop]))
        previous = source, teacher, new
    capture.release()
    native = next(pixels for _, _, pixels in decode_frames(
        args.source_video, indices=[45], hlg_exposure_stops=-1.0))
    capture = cv2.VideoCapture(str(args.new_video))
    native_new = video_frame(capture, 45)
    capture.release()
    teacher = read_rgb(args.teacher_dir / "styled_000045.png")
    teacher_up = cv2.resize(teacher, (1920, 1080),
                            interpolation=cv2.INTER_CUBIC)
    native_crop = np.s_[650:1000, 1300:1830]
    result = {
        "schema": "native-detail-review-v1",
        "method": "Optical flow from source frames, stable pixels with source warp L1 < 18. Right vegetation crop x=500:768,y=170:400 at 768x432. Laplacian energy at native crop x=1300:1830,y=650:1000 on frame45.",
        "sample_indices": indices,
        "frame_transitions": len(all_scores),
        "whole_scene": {key: float(np.mean([r[key] for r in all_scores]))
                        for key in all_scores[0]},
        "grass_crop": {key: float(np.mean([r[key] for r in grass_scores]))
                       for key in grass_scores[0]},
        "native_detail_frame45": {
            "source": focus_energy(native[native_crop]),
            "teacher_768_upscaled": focus_energy(teacher_up[native_crop]),
            "new_1080p": focus_energy(native_new[native_crop]),
        },
    }
    crops = [native[native_crop], teacher_up[native_crop], native_new[native_crop]]
    tile_width = crops[0].shape[1]
    tile_height = crops[0].shape[0]
    board = Image.new("RGB", (tile_width * 3, tile_height + 40), "#202020")
    draw = ImageDraw.Draw(board)
    labels = ("Original 1080p", "Old 768p enlarged", "New 1080p")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    for number, (pixels, label) in enumerate(zip(crops, labels)):
        board.paste(Image.fromarray(pixels), (tile_width * number, 40))
        draw.text((tile_width * number + 10, 7), label,
                  fill="white", font=font)
    comparison = args.output.with_name("native_grass_detail_comparison.jpg")
    board.save(comparison, quality=96)
    result["detail_comparison"] = str(comparison)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
