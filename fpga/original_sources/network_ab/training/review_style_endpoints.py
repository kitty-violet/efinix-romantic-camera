"""Check the few start/end seconds omitted by interval sampling."""
from __future__ import annotations

from pathlib import Path
import json

import imageio_ffmpeg
from PIL import Image, ImageDraw

from .extract_new_style_candidates import decode

ROOT = Path(r"E:\fpga\model-training-workspace")
SOURCE = ROOT / "data/incoming/style-target-replacement"
OUT = ROOT / "runs/style-target-new-sources-20261001/endpoint-review"
SAMPLES = [
    ("Dying Light 2026-10-01 18-54-23.mp4", 1),
    ("Dying Light 2026-10-01 18-54-23.mp4", 172),
    ("Dying Light 2026-10-01 18-59-02.mp4", 1),
    ("Dying Light 2026-10-01 18-59-02.mp4", 2090),
    ("屏幕录制 2026-10-01 194608.mp4", 39),
]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    canvas = Image.new("RGB", (1680, 536), "#171717")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for slot, (name, second) in enumerate(SAMPLES):
        path = OUT / f"endpoint_{slot:02d}.png"
        row = {"source": str(SOURCE / name), "second": second}
        row.update(decode(SOURCE / name, second, path, exe))
        rows.append(row)
        x, y = slot % 4 * 420, slot // 4 * 268
        with Image.open(path) as image:
            canvas.paste(image.resize((420, 236), Image.Resampling.LANCZOS), (x, y + 28))
        draw.text((x + 5, y + 5), f"{name[-18:]} {second}s", fill="white")
    contact = OUT / "endpoints.jpg"
    canvas.save(contact, quality=94)
    (OUT / "index.json").write_text(json.dumps({
        "schema": "new-style-endpoint-review-v1", "status": "REVIEW_ONLY",
        "rows": rows, "contact": str(contact),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"endpoint frames={len(rows)}")


if __name__ == "__main__":
    main()
