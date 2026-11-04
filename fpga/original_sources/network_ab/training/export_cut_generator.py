"""Export a generator-only FP32 state dict from an audited CUT checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .runtime import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["step"] != 2000 or state["config"]["precision"] != "FP32-no-AMP-no-TF32":
        raise ValueError("Expected a completed FP32 2000-step checkpoint")
    weights = state["generator"]
    if not weights or any(value.dtype != torch.float32 for value in weights.values()):
        raise ValueError("Generator weights are missing or not FP32")
    torch.save(weights, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    if weights.keys() != reloaded.keys() or any(
            not torch.equal(value, reloaded[key]) for key, value in weights.items()):
        raise RuntimeError("Exported state dict differs from checkpoint")
    write_json(args.output.with_suffix(".json"), {
        "schema": "cut-generator-fp32-v1", "step": state["step"],
        "generator_arch": state["config"].get("generator_arch", "resnet"),
        "generator_width": state["config"]["generator_width"],
        "residual_blocks": state["config"]["residual_blocks"],
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
        "weights_sha256": file_hash(args.output),
        "tensor_count": len(weights), "parameter_count": sum(v.numel() for v in weights.values()),
        "verified_tensor_equal": True,
        "quality_gate": "EXPERIMENTAL_VISUAL_REVIEW_REQUIRED",
    })
    print(args.output)


if __name__ == "__main__":
    main()
