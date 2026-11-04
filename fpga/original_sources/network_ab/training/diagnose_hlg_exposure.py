"""Compare reproducible HLG-to-SDR exposure variants on one source instant.

This is a diagnostic only. It never edits approved datasets or media policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from .media import HLG_TO_SRGB, probe_video
from .runtime import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = probe_video(args.source)
    if metadata["color_tags"]["color_trc"] != "arib-std-b67":
        raise ValueError("Expected HLG source")
    args.output.mkdir(parents=True)
    variants = {
        "approved_mobius_0EV": HLG_TO_SRGB,
        "mobius_minus1EV": HLG_TO_SRGB.replace(
            "tonemap=tonemap=mobius", "exposure=exposure=-1,tonemap=tonemap=mobius"),
        "mobius_minus2EV": HLG_TO_SRGB.replace(
            "tonemap=tonemap=mobius", "exposure=exposure=-2,tonemap=tonemap=mobius"),
        "reinhard_0EV": HLG_TO_SRGB.replace("tonemap=tonemap=mobius:param=0.3",
                                             "tonemap=tonemap=reinhard:param=0.5"),
    }
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    tiles, rows = [], []
    for name, vf in variants.items():
        path = args.output / f"{name}.png"
        cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss",
               f"{args.seconds:.6f}", "-i", str(args.source), "-map", "0:v:0",
               "-frames:v", "1", "-vf", vf + ",scale=960:540:flags=bicubic",
               "-y", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=90)
        if result.returncode or not path.is_file():
            raise RuntimeError(f"{name}: {result.stderr[-1500:]}")
        im = Image.open(path).convert("RGB")
        ar = np.asarray(im)
        rows.append({"variant": name, "filter": vf, "path": str(path),
                     "sha256": file_hash(path), "mean_rgb": ar.mean(axis=(0, 1)).tolist(),
                     "bright_fraction_245": float((ar.max(axis=2) >= 245).mean()),
                     "clipped_fraction_254": float((ar.max(axis=2) >= 254).mean())})
        tiles.append(im)
    canvas = Image.new("RGB", (1920, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, im) in enumerate(zip(variants, tiles)):
        x, y = (index % 2) * 960, (index // 2) * 560
        canvas.paste(im, (x, y + 20))
        draw.text((x + 8, y + 4), name, fill="black")
    canvas.save(args.output / "contact.png")
    (args.output / "diagnostic.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "seconds": args.seconds,
        "source_sha256": file_hash(args.source), "rows": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": [{k: r[k] for k in
        ("variant", "bright_fraction_245", "clipped_fraction_254")}
        for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
