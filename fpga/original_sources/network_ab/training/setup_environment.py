"""Explicit isolated bootstrap; official sources, resumable and verified wheels."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from .runtime import configure_workspace, file_hash, write_json

WHEELS = {"torch": "torch-2.11.0+cu128-cp310-cp310-win_amd64.whl",
          "torchvision": "torchvision-0.26.0+cu128-cp310-cp310-win_amd64.whl"}
PYPI = ["numpy==2.2.6", "pillow==12.3.0", "imageio-ffmpeg==0.6.0",
        "opencv-python-headless==4.12.0.88"]


def download(url, target, digest, *, prefix=False):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    def matches(path):
        sha = file_hash(path)
        return sha.startswith(digest) if prefix else sha == digest
    if target.exists():
        if not matches(target):
            raise RuntimeError(f"Existing file has wrong SHA256: {target}")
        return file_hash(target)
    partial = target.with_suffix(target.suffix + ".partial")
    print(f"DOWNLOAD {url} -> {partial}", flush=True)
    # Python's verified HTTPS avoids Windows Schannel credential-store coupling.
    # Each retry uses a validated Content-Range; never append a full response.
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "model-training-bootstrap/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server refused valid resume range; partial preserved")
                expected = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
                received = 0
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                        if received % (128 * 1024 * 1024) == 0:
                            print(f"DOWNLOAD_BYTES {target.name} {offset + received}", flush=True)
                if expected is not None and received != expected:
                    raise OSError(f"Short download: {received}/{expected}")
            break
        except (OSError, urllib.error.URLError) as exc:
            print(f"RETRY {attempt + 1}: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(3)
    if not matches(partial):
        raise RuntimeError(f"SHA256 mismatch; untrusted partial retained, NOT installed: {partial}")
    partial.replace(target)
    return file_hash(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = configure_workspace(args.root)
    report = root / "runs/environment"
    report.mkdir(parents=True, exist_ok=True)
    if sys.version_info[:2] != (3, 10) or sys.platform != "win32":
        raise RuntimeError("Locked wheels require Windows x64 Python 3.10")
    python = root / "env/Scripts/python.exe"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(root / "env")], check=True)
    verified, sources = [], []
    for package, filename in WHEELS.items():
        index = f"https://download.pytorch.org/whl/cu128/{package}/"
        page = urllib.request.urlopen(index, timeout=60).read().decode()
        matches = [html.unescape(url) for url in re.findall(r'href="([^"]+)"', page)
                   if urllib.parse.unquote(url.split("#")[0]).endswith(filename)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one official wheel link for {filename}")
        url, fragment = matches[0].split("#", 1)
        if urllib.parse.urlparse(url).hostname not in ("download.pytorch.org", "download-r2.pytorch.org"):
            raise RuntimeError("Unexpected official index download host")
        digest = urllib.parse.parse_qs(fragment)["sha256"][0]
        if not re.fullmatch("[0-9a-f]{64}", digest):
            raise RuntimeError("Missing full official SHA256")
        wheel = root / "cache/wheels" / filename
        actual = download(url, wheel, digest)
        verified.append(str(wheel))
        sources.append({"index": index, "url": url, "file": str(wheel), "sha256": actual})
        write_json(report / "official-wheels.json", sources)
        print(f"VERIFIED {filename} {actual}", flush=True)
    # pip's resolver also verifies index-declared hashes for transitive wheels.
    subprocess.run([str(python), "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                    "--only-binary=:all:", "--timeout", "120", "--retries", "5",
                    "--report", str(report / "pip-install-report.json"), *verified, *PYPI], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (report / "requirements.freeze.txt").write_text(frozen, encoding="utf-8")
    vgg = root / "cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth"
    sha = download("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", vgg, "dcbb9e9d", prefix=True)
    write_json(report / "vgg19-weights.json", {"weights": "VGG19_Weights.IMAGENET1K_V1",
               "source": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
               "path": str(vgg), "official_sha256_prefix": "dcbb9e9d", "sha256": sha})
    print("ENVIRONMENT_PACKAGES_READY; CUDA model validation still required", flush=True)


if __name__ == "__main__":
    main()
