"""Finalize visually reviewed native game references; preserve the v3 draft."""

import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from network_ab.training.runtime import configure_workspace, file_hash, write_json

ROOT = configure_workspace()
SOURCE = ROOT / "data/processed/style-target-20261001-v3-clear-wide"
TARGET = ROOT / "data/processed/style-target-20261001-v4-wide-reviewed"
REVIEW = ROOT / "runs/style-target-full-timeline-20261001"

# Panoramas are the principal reference: native crops preserve spatial context.
# The blue special-vision sequence at 20:25 is excluded from the normal-game style.
WIDE = {
    "2_00495", "2_00965", "3_00225", "3_00555", "3_01215",
    "3_01880", "3_02250", "3_02380", "3_02410", "3_02985", "3_03285",
}
TEXTURE = {
    "1_00065_A", "2_00145_C", "2_00145_D", "2_00470_A",
    "2_00470_D", "2_00575_A", "2_00965_A", "2_00965_D",
    "3_00225_D", "3_00225_F", "3_00280_D", "3_00405_H",
    "3_00440_D", "3_00440_H", "3_00485_E", "3_00565_A",
    "3_00565_H", "3_01150_A", "3_01245_D", "3_01335_F",
    "3_01525_B", "3_01805_A", "3_01880_B", "3_01970_A",
    "3_02250_A", "3_02250_E", "3_02380_A", "3_02380_B",
    "3_02410_C", "3_02480_A", "3_02480_B", "3_03120_B",
    "3_03320_F", "3_03580_A", "3_03580_G",
}


def contact(records, kind, page_size, size, cols):
    items = [r for r in records if r["kind"] == kind]
    paths = []
    for start in range(0, len(items), page_size):
        chunk = items[start:start + page_size]
        rows = (len(chunk) + cols - 1) // cols
        canvas = Image.new("RGB", (size[0] * cols, (size[1] + 24) * rows), "#161616")
        pen = ImageDraw.Draw(canvas)
        for n, row in enumerate(chunk):
            x, y = n % cols * size[0], n // cols * (size[1] + 24)
            with Image.open(TARGET / row["file"]) as im:
                canvas.paste(im.resize(size, Image.Resampling.BICUBIC), (x, y + 24))
            pen.text((x + 4, y + 4), row["file"], fill="white")
        path = REVIEW / f"v4_{kind}_contact_{start // page_size:02d}.jpg"
        canvas.save(path, quality=94)
        paths.append(str(path))
    return paths


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    prior = json.loads((SOURCE / "dataset-manifest.json").read_text(encoding="utf-8"))
    wanted = {"W_" + value + ".png" for value in WIDE}
    wanted |= {"T_" + value + ".png" for value in TEXTURE}
    index = {r["file"]: r for r in prior["records"]}
    if wanted - index.keys():
        raise ValueError(f"Missing reviewed images: {wanted - index.keys()}")
    TARGET.mkdir(parents=True)
    records = []
    for name in sorted(wanted):
        row = dict(index[name])
        source = SOURCE / name
        if file_hash(source) != row["sha256"]:
            raise ValueError(f"Source hash changed: {name}")
        shutil.copy2(source, TARGET / name)
        row["review"] = "manual final contact-sheet review: native detail, low UI, scene diversity"
        records.append(row)
    pages = contact(records, "wide_scene", 12, (440, 247), 3)
    pages += contact(records, "texture_region", 30, (220, 220), 5)
    manifest = {
        "schema": "model-training-style-v1",
        "status": "curated_for_training",
        "image_count": len(records),
        "wide_scene_count": len(WIDE),
        "texture_region_count": len(TEXTURE),
        "source_times": len({(r["source_video"], r["requested_second"]) for r in records}),
        "source_scan": prior["source_scan"],
        "parent_draft": str(SOURCE / "dataset-manifest.json"),
        "sampling_policy": {"wide_scene_probability": 0.65,
                            "texture_region_probability": 0.35},
        "exclusions": ["desktop/menu", "soft-focus and heavy motion blur",
                       "legible subtitle/mission UI", "blue temporary in-game vision filter",
                       "near-duplicate patches from the same frame"],
        "native_frame_wh": prior["native_frame_wh"],
        "records": records,
        "contact_pages": pages,
    }
    write_json(TARGET / "dataset-manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("image_count", "wide_scene_count", "texture_region_count", "source_times")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
