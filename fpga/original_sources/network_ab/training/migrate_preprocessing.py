"""Explicit continuation after the audited pixel-decoder correction, not exact replay.

Keep learned parameters, Adam moments, schedule and global RNG. Require the
same evaluation frames; permit only recorded training-frame exclusions from
input-quality review. Reset the sampler if exclusions change the training pool.
Old/new validation objectives are not directly comparable, so reset its history.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from .runtime import (HERE, assert_fingerprints, atomic_replace, canonical_hash,
    configure_workspace, file_hash, snapshot_sources, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path, required=True)
    parser.add_argument("--real-data-report", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "fp32_recovery_config.json")
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Run must reside in the authorized workspace")
    import torch
    from .data import Sampler, audit_style, load_content_manifest
    old = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    parent, data = load_content_manifest(args.parent_manifest), load_content_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = json.loads(args.gpu_report.read_text(encoding="utf-8"))
    real = json.loads(args.real_data_report.read_text(encoding="utf-8"))
    assert_fingerprints(gate["fingerprints"])
    if (gate["status"] != "PASS_DIAGNOSTIC_ONLY" or real["status"] != "PASS_REAL_CONTENT_RESUME"
            or real["binding"]["fingerprints"] != gate["fingerprints"]
            or real["binding"]["data_manifest_sha256"] != file_hash(args.manifest)
            or gate["config_sha256"] != file_hash(args.config) or old["config"] != config
            or old["purpose"] != "fp32_training"):
        raise ValueError("Mismatched configuration or GPU/real-data gate")
    if (old["binding"]["data_manifest_sha256"] != file_hash(args.parent_manifest)
            or data["parent_manifest_sha256"] != file_hash(args.parent_manifest)):
        raise ValueError("Dataset parent provenance mismatch")
    def selection(rows):
        return [(r["source_sha256"], r["group"], r["split"], Path(r["path"]).name) for r in rows]
    parent_selection, new_selection = selection(parent["images"]), selection(data["images"])
    omissions = [x for x in parent_selection if x not in new_selection]
    recorded = {row["parent_image_sha256"] for row in data.get("quality_exclusions", [])}
    expected_removed = {r["sha256"] for r in parent["images"]
                        if selection([r])[0] in omissions}
    if (any(x[2] != "train" for x in omissions) or recorded != expected_removed
            or [x for x in parent_selection if x not in omissions] != new_selection):
        raise ValueError("Unreviewed source/frame/order/split change")
    prior, now = old["binding"]["fingerprints"], gate["fingerprints"]
    if prior["shared_files"] != now["shared_files"]:
        raise ValueError("Shared contract or model builder changed")
    changed = {p for p in prior["training_files"].keys() | now["training_files"].keys()
               if prior["training_files"].get(p) != now["training_files"].get(p)}
    allowed = {"media.py", "prepare_data.py", "evaluate.py", "review_uploads.py",
               "redecode_data.py", "migrate_preprocessing.py", "tests/test_media.py"}
    if not changed <= allowed:
        raise ValueError(f"Unreviewed training changes during data correction: {sorted(changed-allowed)}")
    style = audit_style()
    if old["binding"]["style_manifest_sha256"] != canonical_hash(style):
        raise ValueError("Style references changed")
    sampler = Sampler([r for r in data["images"] if r["split"] == "train"], style, config)
    new = copy.deepcopy(old)
    new["binding"].update(fingerprints=now, data_manifest_sha256=file_hash(args.manifest),
                          gpu_report_sha256=file_hash(args.gpu_report))
    if omissions:
        new["sampler"] = sampler.state_dict()
    else:
        new["sampler"]["identity"] = sampler.identity
    sampler.load_state_dict(new["sampler"])
    new["control"] = {"best_validation_objective": None, "unimproved_validations": 0}
    args.run.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.run / "source", now)
    checkpoint = args.run / f"corrected_data_start_{old['completed_steps']:06d}.pt"
    with tempfile.NamedTemporaryFile(dir=args.run, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(new, stream)
    atomic_replace(temporary, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    def exact(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                exact(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                exact(x, y)
        else:
            assert a == b
    for key in old.keys() - {"binding", "sampler", "control"}:
        exact(old[key], loaded[key])
    if not omissions:
        for key in old["sampler"].keys() - {"identity"}:
            exact(old["sampler"][key], loaded["sampler"][key])
    report = {"status": "PASS_EXPLICIT_DATA_CORRECTION_CONTINUATION", "parent_checkpoint": str(args.checkpoint),
        "parent_sha256": file_hash(args.checkpoint), "checkpoint": str(checkpoint), "sha256": file_hash(checkpoint),
        "changed_sources": sorted(changed), "evaluation_frame_order_splits_equal": True,
        "excluded_training_frames": len(omissions),
        "model_adam_schedule_calibration_global_rng_exact": True,
        "sampler_policy": "reset from original seed for corrected pool" if omissions else "order/cursor/RNG exact; identity updated",
        "validation_control_reset": True, "style_weight_retained": old["calibration"]["style_weight"],
        "meaning": "New corrected pixels change the subsequent trajectory; this is not exact replay of the legacy dataset",
        "real_data_gate_sha256": file_hash(args.real_data_report), "quantization_allowed": False}
    write_json(args.run / "migration.json", report)
    write_json(args.run / "run.json", {"binding": new["binding"], "config": config,
        "calibration": new["calibration"], "manifest": data, "status": "READY_CORRECTED_DATA_CONTINUATION",
        "precision": "FP32; TF32 and AMP disabled", "migration": report})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
