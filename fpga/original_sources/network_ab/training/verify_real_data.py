"""Actual uploaded-content FP32 short training and uninterrupted/resumed comparison."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

from .runtime import HERE, assert_fingerprints, canonical_hash, configure_workspace, file_hash, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "fp32_config.json")
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "runs").resolve() not in args.output.resolve().parents:
        raise ValueError("Output must reside under workspace/runs")
    import torch
    from .data import Sampler, audit_style, load_content_manifest
    from .engine import make_training, restore_checkpoint, save_checkpoint, set_reproducible, train_step
    from .losses import PerceptualLoss, calibrate_first_batch
    gate = json.loads(args.gpu_report.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_DIAGNOSTIC_ONLY":
        raise ValueError("GPU gate missing")
    assert_fingerprints(gate["fingerprints"])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if gate["config_sha256"] != file_hash(args.config):
        raise ValueError("GPU-tested configuration changed")
    manifest = load_content_manifest(args.manifest)
    style = audit_style()
    records = [r for r in manifest["images"] if r["split"] == "train"]
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output / "source", gate["fingerprints"])
    report = {"status": "RUNNING", "purpose": "REAL_CONTENT_DIAGNOSTIC_NOT_DELIVERY_WEIGHTS",
              "config_sha256": file_hash(args.config)}
    try:
        set_reproducible(config["seed"])
        model, optimizer, scheduler = make_training(config)
        sampler = Sampler(records, style, config)
        criterion = PerceptualLoss(root).cuda()
        binding = {"fingerprints": gate["fingerprints"], "data_manifest_sha256": file_hash(args.manifest),
                   "style_manifest_sha256": canonical_hash(style), "vgg_sha256": criterion.weights_sha256}
        before = sampler.state_dict()
        before_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        before_modes = [m.training for m in model.modules()]
        before_rng = (random.getstate(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()])
        calibration = calibrate_first_batch(model, criterion, sampler, config)
        assert before == sampler.state_dict()
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before_model.items())
        assert before_modes == [m.training for m in model.modules()]
        assert random.getstate() == before_rng[0] and torch.equal(torch.get_rng_state(), before_rng[1])
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), before_rng[2]))
        assert abs(calibration["achieved_ratio"]-config["style_content_gradient_ratio"]) < 1e-12
        report.update(binding=binding, calibration=calibration, calibration_state_restoration_checked=True)
        report["short_training"] = [train_step(model, optimizer, scheduler, criterion, sampler, s, calibration["style_weight"], config) for s in range(2)]
        checkpoint = args.output / "diagnostic_real_content_step2.pt"
        report["checkpoint_sha256"] = save_checkpoint(checkpoint, model, optimizer, scheduler, sampler,
            step=2, binding=binding, config=config, calibration=calibration, purpose="diagnostic_only")
        expected_metrics = train_step(model, optimizer, scheduler, criterion, sampler, 2, calibration["style_weight"], config)
        expected_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler, expected_sampler = copy.deepcopy(scheduler.state_dict()), sampler.state_dict()
        expected_rng = (random.getstate(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()])
        restored, opt2, sched2 = make_training(config)
        sampler2 = Sampler(records, style, config)
        restore_checkpoint(checkpoint, restored, opt2, sched2, sampler2, binding=binding, config=config, purpose="diagnostic_only")
        actual_metrics = train_step(restored, opt2, sched2, criterion, sampler2, 2, calibration["style_weight"], config)
        def compare(a, b):
            if torch.is_tensor(a):
                return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max()) if a.numel() else 0.0
            if isinstance(a, dict):
                assert a.keys() == b.keys()
                return max((compare(a[k], b[k]) for k in a), default=0.0)
            if isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                return max((compare(x, y) for x, y in zip(a, b)), default=0.0)
            assert a == b
            return 0.0
        model_error = compare(expected_model, restored.state_dict())
        optimizer_error = compare(expected_optimizer, opt2.state_dict())
        assert model_error <= 1e-6 and optimizer_error <= 1e-6
        assert expected_scheduler == sched2.state_dict() and expected_sampler == sampler2.state_dict()
        assert expected_metrics["samples"] == actual_metrics["samples"]
        assert expected_rng[0] == random.getstate()
        assert torch.equal(expected_rng[1], torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(expected_rng[2], torch.cuda.get_rng_state_all()))
        report.update(status="PASS_REAL_CONTENT_RESUME", model_max_absolute_error=model_error,
                      optimizer_max_absolute_error=optimizer_error, scheduler_sampler_rng_equal=True,
                      uninterrupted_step3=expected_metrics, resumed_step3=actual_metrics,
                      precision="FP32; AMP/TF32 disabled", quantization_allowed=False)
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "model_max_absolute_error", "optimizer_max_absolute_error")}))


if __name__ == "__main__":
    main()
