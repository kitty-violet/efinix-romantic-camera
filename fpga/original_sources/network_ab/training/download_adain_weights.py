"""Fetch two documented AdaIN release tensors into the isolated model cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path("E:/fpga/model-training-workspace/cache/pretrained/adain")
BASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/"
FILES = ("vgg_normalised.pth", "decoder.pth")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        destination = ROOT / name
        temporary = ROOT / (name + ".part")
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(BASE + name, timeout=60) as response, temporary.open("wb") as output:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {name}")
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size < 10_000_000:
            raise RuntimeError(f"Short release asset: {name}, {size} bytes")
        temporary.replace(destination)
        records.append({"file": str(destination), "source": BASE + name,
                        "bytes": size, "sha256": digest.hexdigest()})
        print(f"verified transfer {name}: {size} bytes, sha256={digest.hexdigest()}", flush=True)
    (ROOT / "sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
