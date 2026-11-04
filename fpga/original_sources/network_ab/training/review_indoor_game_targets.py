"""Display all previously extracted indoor game patches before any selection."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .runtime import configure_workspace, file_hash, write_json


def main() -> None:
    root = configure_workspace()
    review = root / "runs/style-target-full-timeline-20261001/center_review.json"
    data = json.loads(review.read_text(encoding="utf-8"))
    if data.get("schema") is None or data.get("status") != "REVIEW_ONLY":
        raise ValueError("Indoor patch source drift")
    rows = [row for row in data["patches"] if row.get("scene") == "cool_indoor"]
    if len(rows) != 36:
        raise ValueError(f"Expected 36 indoor patches, found {len(rows)}")
    out = root / "runs/material-safe-720-20261002/indoor-target-review"
    out.mkdir(parents=True, exist_ok=True)
    board = Image.new("RGB", (6 * 300, 6 * 320), "#151515")
    draw = ImageDraw.Draw(board)
    for i, row in enumerate(rows):
        path = Path(row["path"])
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Indoor patch changed: {path}")
        image = Image.open(path).convert("RGB")
        if image.size != (512, 512):
            raise ValueError(f"Wrong patch size: {path}")
        x, y = (i % 6) * 300, (i // 6) * 320
        board.paste(image.resize((300, 300), Image.Resampling.LANCZOS), (x, y + 20))
        draw.text((x + 5, y + 4), row["id"], fill="white")
    contact = out / "all_36_indoor_patches.jpg"
    board.save(contact, quality=92)
    write_json(out / "candidates.json", {
        "schema": "indoor-game-patch-review-v1", "status": "visual_review_required",
        "source_review": str(review), "source_review_sha256": file_hash(review),
        "count": len(rows), "records": rows,
        "contact_sheet": str(contact), "contact_sheet_sha256": file_hash(contact)})
    print(contact)


if __name__ == "__main__":
    main()
