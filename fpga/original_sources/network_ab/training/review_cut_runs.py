"""Record visual quality decisions for finished CUT diagnostic checkpoints."""
from __future__ import annotations

import json

from .runtime import configure_workspace, file_hash, write_json


REVIEWS = {
    "cut-wide-20261001-02": (500, "Severe orange/green cast and false high-frequency texture"),
    "cut-wide-20261001-03": (1000, "Neon yellow/green highlights and false tree/pavement texture"),
    "cut-wide-20261001-04-exposure": (500, "Exposure improved input, but orange sky and coarse texture persist"),
    "cut-wide-20261001-05-daylight": (500, "Orange cast reduced; cyan sky, dirty granular surfaces and inadequate game material"),
    "cut-wide-20261001-06-residual": (500, "Geometry preserved but mostly warm tint, with insufficient game-model appearance"),
    "cut-wide-20261001-07-vgg": (500, "Some blur reduced; cyan sky, sign color error and inadequate game material"),
}


def main():
    root = configure_workspace() / "runs"
    for name, (step, reason) in REVIEWS.items():
        folder = root / name
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        checkpoint = folder / f"checkpoint_{step:06d}.pt"
        comparison = folder / f"validation_{step:06d}/comparison_02.png"
        if status["step"] != step or status["status"] != "STOP_AFTER_REACHED":
            raise ValueError(f"Unexpected run state: {name}")
        if not checkpoint.is_file() or not comparison.is_file():
            raise FileNotFoundError(f"Missing review evidence: {name}")
        status.update(quality_gate="FAILED_VISUAL_REVIEW", reason=reason,
            review_checkpoint_sha256=file_hash(checkpoint),
            review_comparison_sha256=file_hash(comparison),
            delivery_policy="Diagnostic evidence only; not a qualified style model")
        write_json(status_path, status)
        print(name, step, "FAILED_VISUAL_REVIEW")


if __name__ == "__main__":
    main()
