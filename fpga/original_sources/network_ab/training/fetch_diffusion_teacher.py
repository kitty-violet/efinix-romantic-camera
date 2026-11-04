"""Fetch pinned, safetensors-only SD1.5/ControlNet components into the project cache."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .runtime import configure_workspace, file_hash, write_json


SOURCES = (
    ("sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5",
     "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
     ["model_index.json", "scheduler/scheduler_config.json", "tokenizer/*",
      "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
      "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
      "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
      "feature_extractor/preprocessor_config.json"]),
    ("controlnet_canny", "lllyasviel/control_v11p_sd15_canny",
     "115a470d547982438f70198e353a921996e2e819",
     ["config.json", "diffusion_pytorch_model.fp16.safetensors"]),
    ("ip_adapter_plus", "h94/IP-Adapter",
     "018e402774aeeddd60609b4ecdb7e298259dc729",
     ["models/image_encoder/config.json",
      "models/image_encoder/model.safetensors",
      "models/ip-adapter-plus_sd15.safetensors"]),
    ("instruct_pix2pix", "timbrooks/instruct-pix2pix",
     "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d",
     ["model_index.json", "scheduler/scheduler_config.json", "tokenizer/*",
      "feature_extractor/preprocessor_config.json",
      "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
      "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
      "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[source[0] for source in SOURCES])
    args = parser.parse_args()
    root = configure_workspace()
    cache = root / "cache/huggingface/hub"
    report_dir = root / "runs/diffusion-teacher-preflight"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, repo_id, revision, patterns in SOURCES:
        if args.only and name != args.only:
            continue
        folder = Path(snapshot_download(repo_id=repo_id, revision=revision,
                                        cache_dir=cache, allow_patterns=patterns,
                                        max_workers=2))
        files = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in (".bin", ".ckpt", ".pth", ".py"):
                raise ValueError(f"Unexpected executable or pickle asset: {path}")
            files.append({"relative_path": str(path.relative_to(folder)).replace("\\", "/"),
                          "bytes": path.stat().st_size, "sha256": file_hash(path)})
        if not files:
            raise ValueError(f"Empty model snapshot: {name}")
        write_json(report_dir / f"{name}.json", {
            "source": "huggingface.co", "repo_id": repo_id, "revision": revision,
            "snapshot": str(folder), "files": files,
            "total_bytes": sum(row["bytes"] for row in files),
        })
        print(name, folder, sum(row["bytes"] for row in files))


if __name__ == "__main__":
    main()
