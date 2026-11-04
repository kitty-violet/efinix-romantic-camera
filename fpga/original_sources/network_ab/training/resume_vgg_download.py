"""Resume the official VGG file with bounded, validated HTTP byte ranges."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import urllib.request

from .runtime import configure_workspace, file_hash, write_json

URL = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"


def main():
    root = configure_workspace()
    target = root / "cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth"
    partial = target.with_suffix(".pth.partial")
    manifest = root / "runs/environment/vgg19-weights.json"
    if target.exists():
        digest = file_hash(target)
        if not digest.startswith("dcbb9e9d"):
            raise RuntimeError("Existing VGG hash mismatch")
    else:
        request = urllib.request.Request(URL, method="HEAD", headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers["Content-Length"])
            etag = response.headers["ETag"]
        if total != 574673361:
            raise RuntimeError("Official VGG object length changed")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > total:
            raise RuntimeError("Partial file is larger than official object")
        directory = root / "tmp/vgg-http-ranges"
        directory.mkdir(parents=True, exist_ok=True)
        ranges = [(start, min(total-1, start+32*1024*1024-1))
                  for start in range(offset, total, 32*1024*1024)]
        print(f"RESUME_OFFICIAL offset={offset} total={total} parts={len(ranges)} workers=4", flush=True)

        def fetch(bounds):
            first, last = bounds
            part = directory / f"bytes-{first}-{last}.part"
            expected_size = last-first+1
            for attempt in range(6):
                have = part.stat().st_size if part.exists() else 0
                if have == expected_size:
                    return part
                if have > expected_size:
                    raise RuntimeError("Oversized range file")
                start = first+have
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{last}",
                           "If-Match": etag, "Accept-Encoding": "identity"})
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{last}/{total}"
                                or response.headers.get("ETag") != etag):
                            raise RuntimeError("Server changed object or returned an incorrect byte range")
                        with part.open("ab") as stream:
                            while block := response.read(1024*1024):
                                stream.write(block)
                    if part.stat().st_size != expected_size:
                        raise OSError("Short range response")
                    print(f"RANGE_COMPLETE {first}-{last}", flush=True)
                    return part
                except OSError as exc:
                    if attempt == 5:
                        raise
                    print(f"RANGE_RETRY {first}-{last}: {exc}", flush=True)
                    time.sleep(2)
            raise RuntimeError("Unreachable range state")

        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(fetch, ranges))
        # The old sequential downloader must be stopped before running this tool.
        if (partial.stat().st_size if partial.exists() else 0) != offset:
            raise RuntimeError("Another writer changed the partial; merge refused")
        with partial.open("ab") as destination:
            for part in parts:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(4*1024*1024), b""):
                        destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        if partial.stat().st_size != total:
            raise RuntimeError("Merged file length mismatch")
        digest = file_hash(partial)
        if not digest.startswith("dcbb9e9d"):
            raise RuntimeError("Merged VGG SHA256 mismatch; retained, never loaded")
        partial.replace(target)
        write_json(root / "runs/environment/vgg-download-ranges.json",
                   {"url": URL, "etag": etag, "total_bytes": total, "resumed_bytes": offset,
                    "ranges": ranges, "sha256": digest, "workers": 4})
    write_json(manifest, {"weights": "VGG19_Weights.IMAGENET1K_V1", "source": URL,
               "path": str(target), "official_sha256_prefix": "dcbb9e9d", "sha256": digest})
    print(f"VGG_VERIFIED sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
