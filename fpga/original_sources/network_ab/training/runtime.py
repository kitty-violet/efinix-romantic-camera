"""Paths, contract drift guard, atomic JSON and reproducibility fingerprints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from network_ab.model_contract import canonical_hash, compile_contract, load_contract

HERE = Path(__file__).resolve().parent
NETWORK = HERE.parent
PROJECT = NETWORK.parent
EXPECTED_CONTRACT = "b330b3bce8d09544d596ff446a457c60c77407ded12800cd9cceb16fba92abf3"
SHARED = ("model_b_contract.json", "model_contract.schema.json", "model_contract.py",
          "contract_torch.py", "contract_reference.py")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_replace(source, destination):
    """Retry transient Windows read/share locks without falling back to torn writes."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if attempt == 9 or getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(min(0.02 * 2**attempt, 0.5))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-volume atomic replacement; Unicode Windows paths are supported.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    atomic_replace(temporary, path)


def configure_workspace(root=None):
    config = json.loads((HERE / "workspace.json").read_text(encoding="utf-8"))
    root = Path(root or config["root"]).resolve()
    values = {"TEMP": root / "tmp", "TMP": root / "tmp",
              "PIP_CACHE_DIR": root / "cache/pip", "TORCH_HOME": root / "cache/torch",
              "CUDA_CACHE_PATH": root / "cache/cuda", "XDG_CACHE_HOME": root / "cache",
              "TORCHINDUCTOR_CACHE_DIR": root / "cache/inductor",
              "TRITON_CACHE_DIR": root / "cache/triton"}
    for key, value in values.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    tempfile.tempdir = str(root / "tmp")
    return root


def contract_guard(expected=EXPECTED_CONTRACT):
    contract = load_contract()
    plan = compile_contract(contract)
    if plan["contract_sha256"] != expected:
        raise RuntimeError("Shared contract changed: coordinate with user before training/resume/export")
    return contract, plan


def fingerprints():
    _, plan = contract_guard()
    return {"contract_canonical_sha256": plan["contract_sha256"],
            "shared_files": {p: file_hash(NETWORK / p) for p in SHARED},
            "training_files": {p.relative_to(HERE).as_posix(): file_hash(p)
                               for p in sorted(HERE.rglob("*"))
                               if p.is_file() and p.suffix in (".py", ".json", ".ps1")}}


def assert_fingerprints(expected):
    actual = fingerprints()
    if actual != expected:
        raise RuntimeError("Contract or training source/config drift: refusing old run")
    return canonical_hash(actual)


def snapshot_sources(destination, expected):
    """Freeze the actual Python/JSON sources, not merely their now-live paths."""
    assert_fingerprints(expected)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in expected["shared_files"]:
        target = destination / "network_ab" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NETWORK / name, target)
        if file_hash(target) != expected["shared_files"][name]:
            raise RuntimeError("Shared source changed while snapshotting")
    for name in expected["training_files"]:
        target = destination / "network_ab/training" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
        if file_hash(target) != expected["training_files"][name]:
            raise RuntimeError("Training source changed while snapshotting")
    assert_fingerprints(expected)
    write_json(destination / "source-manifest.json", expected)
