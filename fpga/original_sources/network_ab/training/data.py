"""Audited manifests and serial, exactly resumable sampling; no auto split leakage."""
from __future__ import annotations

import json
from pathlib import Path
import random

from .runtime import file_hash, canonical_hash

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
WORKSPACE = json.loads((Path(__file__).resolve().parent / "workspace.json").read_text(encoding="utf-8"))
STYLE_ROOT = Path(WORKSPACE["style_reference"])


def audit_style(root=STYLE_ROOT):
    root = Path(root)
    if any(root.resolve() == Path(old).resolve() for old in WORKSPACE.get("retired_style_references", [])):
        raise RuntimeError(f"Target style set was retired by user: {root}")
    if root.resolve() == STYLE_ROOT.resolve() and WORKSPACE.get("style_reference_status") != "approved":
        raise RuntimeError(f"Target style set is retired/pending approval: {root}")
    manifest = json.loads((root / "dataset-manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    if manifest.get("status") != "approved":
        raise ValueError("Target style manifest has not been approved")
    if not records or manifest["image_count"] != len(records):
        raise ValueError("Target style manifest count mismatch or empty set")
    output = []
    for row in records:
        if row.get("role") != "style_only":
            raise ValueError("Target style record must be style-only")
        path = (root / row["file"]).resolve()
        if path.parent != root.resolve() or file_hash(path) != row["sha256"]:
            raise ValueError(f"Style file escaped directory or hash changed: {path}")
        output.append({"path": str(path), "sha256": row["sha256"], "role": "style_only"})
    actual = {p.name for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    if actual != {r["file"] for r in records}:
        raise ValueError("Unmanifested/missing style images")
    return output


def audit_incoming(root):
    result = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            continue
        result.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                       "sha256": file_hash(path),
                       "kind": "video" if path.suffix.lower() in VIDEO_SUFFIXES else "image"})
    return result


def load_content_manifest(path, *, check_files=True):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "model-training-data-v1" or manifest.get("status") != "approved":
        raise ValueError("Content manifest must pass source audit and be approved")
    rows = manifest.get("images", [])
    groups, hashes, sources, paths = {}, {}, {}, set()
    for row in rows:
        if row["split"] not in ("train", "val", "test") or row.get("role") != "content":
            raise ValueError("Expected content-only train/val/test records")
        if [row["height"], row["width"]] != [720, 1280]:
            raise ValueError("Content masters must be complete 1280x720 RGB images")
        for key, registry in (("group", groups), ("sha256", hashes), ("source_sha256", sources)):
            identity = row[key]
            if not identity or (identity in registry and registry[identity] != row["split"]):
                raise ValueError(f"Empty grouping or cross-split leakage: {key}={identity}")
            registry[identity] = row["split"]
        resolved = Path(row["path"]).resolve()
        if str(resolved).casefold() in paths or STYLE_ROOT.resolve() in resolved.parents:
            raise ValueError("Duplicate path or style image used as content")
        paths.add(str(resolved).casefold())
        if check_files and file_hash(resolved) != row["sha256"]:
            raise ValueError(f"Content file hash changed: {resolved}")
    if {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("All three nonempty source-disjoint splits required")
    for video in manifest.get("videos", []):
        if video["split"] not in ("val", "test"):
            raise ValueError("Fixed evaluation videos must be val/test only")
        for key, registry in (("group", groups), ("sha256", sources)):
            if not video[key] or (video[key] in registry and registry[video[key]] != video["split"]):
                raise ValueError("Evaluation video source overlaps another split")
            registry[video[key]] = video["split"]
        if check_files and file_hash(video["path"]) != video["sha256"]:
            raise ValueError("Evaluation video hash changed")
    if not any(v["split"] == "test" for v in manifest.get("videos", [])):
        raise ValueError("At least one fixed, unseen test video required")
    return manifest


def image_tensor(path, device="cpu"):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"), copy=True)
    return ((torch.from_numpy(pixels).permute(2, 0, 1).float() - 128.0) / 128.0).to(device)


class Sampler:
    """One serial producer. State includes permutation, cursor and augmentation RNG."""
    def __init__(self, records, style, config):
        self.records, self.style, self.config = records, style, config
        if not records or not style:
            raise ValueError("Empty sampler")
        self.rng = random.Random(config["seed"])
        self.order = list(range(len(records)))
        self.rng.shuffle(self.order)
        self.cursor = self.epoch = 0
        self.identity = canonical_hash({"records": records, "style": style})

    def state_dict(self):
        return {"rng": self.rng.getstate(), "order": self.order[:], "cursor": self.cursor,
                "epoch": self.epoch, "identity": self.identity}

    def load_state_dict(self, state):
        if state["identity"] != self.identity or sorted(state["order"]) != list(range(len(self.records))):
            raise ValueError("Sampler dataset/order mismatch")
        if not 0 <= state["cursor"] <= len(self.records):
            raise ValueError("Sampler cursor outside dataset")
        self.rng.setstate(state["rng"])
        self.order, self.cursor, self.epoch = state["order"][:], state["cursor"], state["epoch"]

    def next_spec(self, step):
        if self.cursor == len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
            self.epoch += 1
        content = self.records[self.order[self.cursor]]
        self.cursor += 1
        full = self.config["full_frame_every"] > 0 and (step + 1) % self.config["full_frame_every"] == 0
        h, w = [720, 1280] if full else self.rng.choices(self.config["crop_hw"], self.config["crop_probabilities"])[0]
        top, left = self.rng.randint(0, 720-h), self.rng.randint(0, 1280-w)
        return {"content": content, "style": self.rng.choice(self.style),
                "crop": [top, left, h, w],
                "flip": self.rng.random() < self.config["horizontal_flip_probability"],
                "style_flip": self.rng.random() < 0.5}

    def next(self, step, device):
        spec = self.next_spec(step)
        image = image_tensor(spec["content"]["path"])
        if list(image.shape) != [3, 720, 1280]:
            raise ValueError("Actual image disagrees with 720p manifest")
        y, x, h, w = spec["crop"]
        image = image[:, y:y+h, x:x+w]
        style = image_tensor(spec["style"]["path"])
        if spec["flip"]:
            image = image.flip(-1)
        if spec["style_flip"]:
            style = style.flip(-1)
        return image.unsqueeze(0).to(device), style.unsqueeze(0).to(device), spec
