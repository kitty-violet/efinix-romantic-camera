"""Attribute sharpness loss to resize, styling, or video compression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .render_material_safe_720 import fit_parameters, process_frame_720, video_frames
from .render_uniform_material_720 import simplify_materials
from .runtime import file_hash, write_json


REGIONS_720 = {"person": (465, 350, 675, 720),
               "marble": (840, 180, 1260, 645)}


def highpass_rms(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return float(np.sqrt(np.mean(np.square(gray - base))))


def crop(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = region
    return image[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--styled", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.styled.parent / "status.json").read_text(encoding="utf-8"))
    params = json.loads((args.styled.parent / "parameters.json").read_text(encoding="utf-8"))
    if (status["video_sha256"] != file_hash(args.styled) or
            status["source_sha256"] != file_hash(args.source)):
        raise ValueError("Render or source changed")
    fitted = fit_parameters(args.source, args.selection)
    if params["source_l_quantiles"] != fitted["source_l_quantiles"]:
        raise ValueError("Style fit changed")
    capture = cv2.VideoCapture(str(args.styled))
    chosen = {0, 75, 149}
    rows = []
    for index, source in video_frames(args.source):
        ok, encoded = capture.read()
        if not ok:
            raise ValueError("Styled video ended early")
        if index not in chosen:
            continue
        down = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
        downup = cv2.resize(down, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        toned = process_frame_720(source, params, params["tone_strength"],
                                  person_weight=1.0)
        simplified = simplify_materials(toned, params["texture_amount"])
        toneup = cv2.resize(toned, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        before_encode = cv2.resize(simplified, (1920, 1080),
                                   interpolation=cv2.INTER_LANCZOS4)
        regions = {}
        for name, region in REGIONS_720.items():
            full = tuple(int(v * 1.5) for v in region)
            images = {"source": crop(source, full),
                      "resize_only": crop(downup, full),
                      "tone_only": crop(toneup, full),
                      "texture_simplified": crop(before_encode, full),
                      "encoded": crop(encoded, full)}
            regions[name] = {"highpass_rms": {k: highpass_rms(v)
                                               for k, v in images.items()},
                             "encode_psnr_db": float(cv2.PSNR(
                                 images["texture_simplified"], images["encoded"])),
                             "encode_mean_absolute_error": float(np.abs(
                                 images["texture_simplified"].astype(np.float32) -
                                 images["encoded"].astype(np.float32)).mean())}
        rows.append({"frame": index, "regions": regions})
    capture.release()
    result = {"schema": "uniform-material-blur-audit-v1",
              "source": str(args.source), "source_sha256": file_hash(args.source),
              "styled": str(args.styled), "styled_sha256": file_hash(args.styled),
              "source_approx_mbps": args.source.stat().st_size * 8 / 5 / 1e6,
              "styled_approx_mbps": args.styled.stat().st_size * 8 / 5 / 1e6,
              "encoder": "H.264 libx264 CRF16 1080p30 yuv420p",
              "frames": rows,
              "interpretation": "PSNR measures compression against reconstructed pre-encode pixels; highpass values indicate relative detail loss. Neither proves perceptual style or all-scene quality."}
    write_json(args.styled.parent / "blur_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
