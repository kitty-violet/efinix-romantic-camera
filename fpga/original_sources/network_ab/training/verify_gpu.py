"""Real CUDA checks using explicit diagnostic patterns, never training content."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
import warnings

from .runtime import HERE, canonical_hash, configure_workspace, file_hash, fingerprints, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "fp32_config.json")
    args = parser.parse_args()
    root = configure_workspace()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torchvision
    from network_ab.model_contract import compile_contract, load_contract
    from .data import audit_style, image_tensor
    from .engine import make_training, restore_checkpoint, save_checkpoint, set_reproducible, train_step
    from .losses import PerceptualLoss, calibrate_weight, gram, rgb01
    config = json.loads(args.config.read_text(encoding="utf-8"))
    binding = {"fingerprints": fingerprints(), "data_manifest_sha256": "DIAGNOSTIC_PATTERNS_NOT_CONTENT"}
    snapshot_sources(args.output / "source", binding["fingerprints"])
    report = {"status": "RUNNING", "purpose": "diagnostic_only; not real content training or quality evidence",
              "fingerprints": binding["fingerprints"], "config_sha256": file_hash(args.config),
              "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
              "training_sizes": {}, "warnings": []}
    captured = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        report["gpu"] = torch.cuda.get_device_name()
        report["device_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        set_reproducible(config["seed"])
        report["precision"] = {"amp": False, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_tf32": torch.backends.cudnn.allow_tf32}
        model, optimizer, scheduler = make_training(config)
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters())
        assert report["trainable_parameters"] == 11379
        style = audit_style()
        criterion = PerceptualLoss(root).cuda()
        binding["vgg_sha256"] = criterion.weights_sha256
        report["vgg_sha256"] = criterion.weights_sha256
        assert not any(p.requires_grad for p in criterion.parameters())
        endpoints = torch.tensor([-1., 0., 127/128], device="cuda")
        torch.testing.assert_close(rgb01(endpoints), torch.tensor([0., 128/255, 1.], device="cuda"), atol=0, rtol=0)
        fixed = torch.arange(24, dtype=torch.float32, device="cuda").reshape(1, 2, 3, 4)
        expected_gram = torch.einsum("nchw,ndhw->ncd", fixed, fixed)/24
        torch.testing.assert_close(gram(fixed), expected_gram, rtol=0, atol=0)
        small = torch.randn(1, 3, 32, 64, device="cuda").clamp(-1, 127/128)
        with torch.no_grad():
            model.eval()
            values = model(small, return_tensors=True)
            for node in compile_contract(load_contract(), input_shape=[32, 64, 3])["nodes"]:
                h, w, c = node["output_shape_hwc"]
                assert list(values[node["id"]].shape) == [1, c, h, w]
                if node["kind"] == "nearest_x2":
                    expected = values[node["inputs"][0]].repeat_interleave(2, -2).repeat_interleave(2, -1)
                    assert torch.equal(values[node["id"]], expected)
                elif node["kind"] == "residual_add":
                    assert torch.equal(values[node["id"]], values[node["inputs"][0]]+values[node["inputs"][1]])
            assert torch.equal(model(small), model(small))
        del values, small
        report["all_29_shapes_nn_residual_repeat_inference"] = True
        # Asymmetric kernel isolates the exact stride2 origin and replicate edge.
        probe = copy.deepcopy(model.operators[model.names["down1.dw"]]).eval()
        convolution = next(layer for layer in probe if isinstance(layer, torch.nn.Conv2d))
        batch_norm = next(layer for layer in probe if isinstance(layer, torch.nn.BatchNorm2d))
        with torch.no_grad():
            convolution.weight.zero_()
            convolution.weight[:, 0, 0, 0] = 1
            batch_norm.weight.fill_(1)
            batch_norm.bias.zero_()
            batch_norm.running_mean.zero_()
            batch_norm.running_var.fill_(1-batch_norm.eps)
            image = torch.arange(1, 33, dtype=torch.float32, device="cuda").reshape(1, 1, 4, 8).repeat(1, 8, 1, 1)
            expected = torch.tensor([[1, 2, 4, 6], [9, 10, 12, 14]], dtype=torch.float32, device="cuda").reshape(1, 1, 2, 4).repeat(1, 8, 1, 1)
            torch.testing.assert_close(probe(image), expected, atol=1e-6, rtol=1e-6)
        report["replicate_stride2_origin_probe"] = True
        del probe, convolution, batch_norm, image, expected

        class DiagnosticSampler:
            def __init__(self, h=64, w=96):
                self.cursor, self.h, self.w = 0, h, w

            def state_dict(self):
                return {"cursor": self.cursor, "height": self.h, "width": self.w, "diagnostic_only": True}

            def load_state_dict(self, state):
                if (state["height"], state["width"]) != (self.h, self.w):
                    raise ValueError("Diagnostic sampler shape changed")
                self.cursor = state["cursor"]

            def next(self, step, device):
                generator = torch.Generator().manual_seed(config["seed"]+self.cursor)
                content = (torch.randint(0, 256, (1, 3, self.h, self.w), generator=generator).float()-128)/128
                target = style[self.cursor % len(style)]
                spec = {"diagnostic_index": self.cursor, "height": self.h, "width": self.w,
                        "style_sha256": target["sha256"]}
                self.cursor += 1
                return content.to(device), image_tensor(target["path"], device).unsqueeze(0), spec

        sampler = DiagnosticSampler()
        content, target_style, spec = sampler.next(0, "cuda")
        model.eval()
        with torch.no_grad():
            output = model(content)
        calibration = calibrate_weight(criterion, output, criterion.targets(content, target_style))
        report["gradient_calibration"] = calibration
        del content, output, target_style
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for step in range(2):
                train_step(model, optimizer, scheduler, criterion, sampler, step, calibration["style_weight"], config)
            report["model_changed_after_training"] = any(not torch.equal(v, model.state_dict()[k]) for k, v in initial.items())
            assert report["model_changed_after_training"]
            checkpoint = args.output / "DIAGNOSTIC_ONLY_checkpoint.pt"
            report["checkpoint_sha256"] = save_checkpoint(checkpoint, model, optimizer, scheduler, sampler,
                step=2, binding=binding, config=config, calibration=calibration, purpose="diagnostic_only")
            uninterrupted = train_step(model, optimizer, scheduler, criterion, sampler, 2, calibration["style_weight"], config)
            expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = copy.deepcopy(scheduler.state_dict())
            expected_sampler = sampler.state_dict()
            expected_rng = torch.rand(8, device="cuda")
            expected_cpu_rng = torch.rand(8)
            expected_python_rng = random.random()
            restored, opt2, sched2 = make_training(config)
            sampler2 = DiagnosticSampler()
            restore_checkpoint(checkpoint, restored, opt2, sched2, sampler2,
                       binding=binding, config=config, purpose="diagnostic_only")
            resumed = train_step(restored, opt2, sched2, criterion, sampler2, 2, calibration["style_weight"], config)
            assert uninterrupted["samples"] == resumed["samples"]
            assert scheduler.state_dict() == expected_scheduler == sched2.state_dict()
            assert expected_sampler == sampler2.state_dict()
            assert torch.equal(expected_rng, torch.rand(8, device="cuda"))
            assert torch.equal(expected_cpu_rng, torch.rand(8))
            assert expected_python_rng == random.random()
            max_error = 0.0
            for key, value in restored.state_dict().items():
                observed = value.detach().cpu()
                torch.testing.assert_close(observed, expected[key], atol=1e-6, rtol=1e-5)
                max_error = max(max_error, (observed.double()-expected[key].double()).abs().max().item())
            max_optimizer_error = 0.0
            for key, state in opt2.state_dict()["state"].items():
                for name, value in state.items():
                    reference = expected_optimizer["state"][key][name]
                    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
                    max_optimizer_error = max(max_optimizer_error, (value.double()-reference.double()).abs().max().item())
            report["resume"] = {"passed": True, "model_max_absolute_error": max_error,
                    "optimizer_max_absolute_error": max_optimizer_error,
                    "tolerance_atol": 1e-6, "tolerance_rtol": 1e-5,
                    "sampler_scheduler_python_cpu_cuda_rng_exact": True,
                    "uninterrupted_content_loss": uninterrupted["content_loss"], "resumed_content_loss": resumed["content_loss"]}
            rejected = False
            try:
                restore_checkpoint(checkpoint, restored, opt2, sched2, sampler2,
                        binding=binding, config=config, purpose="fp32_training")
            except ValueError:
                rejected = True
            assert rejected
            report["diagnostic_checkpoint_rejected_as_formal"] = True
            del restored, opt2, sched2, expected_optimizer, initial
            # True full-resolution student inference, independent of crop training.
            model.eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                image = torch.zeros(1, 3, 720, 1280, device="cuda")
                output = model(image)
                assert list(output.shape) == [1, 3, 720, 1280] and torch.isfinite(output).all()
            torch.cuda.synchronize()
            report["full_720p_inference"] = {"passed": True, "seconds": time.perf_counter()-started,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            del image, output
            for h, w in ([360, 640], [540, 960], [720, 1280]):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    # Four genuine backward passes, one Adam update, effective batch4.
                    metrics = train_step(model, optimizer, scheduler, criterion,
                             DiagnosticSampler(h, w), 3, calibration["style_weight"], config)
                    torch.cuda.synchronize()
                    report["training_sizes"][f"{h}x{w}"] = {"passed": True,
                        "seconds_per_effective_batch4": time.perf_counter()-started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "gradient_l2": metrics["gradient_l2"]}
                except torch.cuda.OutOfMemoryError as exc:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    report["training_sizes"][f"{h}x{w}"] = {"passed": False, "reason": str(exc)}
                write_json(args.output / "progress.json", report)
        # Exercise actual picture/video exporters without presenting synthetic
        # diagnostic outputs as trained visual results.
        from .evaluate import render_video, save_comparison
        import imageio_ffmpeg
        import numpy as np
        sampler = DiagnosticSampler()
        content, _, _ = sampler.next(0, "cuda")
        model.eval()
        with torch.no_grad():
            output = model(content)
        save_comparison(content[0], output[0], args.output / "DIAGNOSTIC_ONLY_picture.png")
        source = args.output / "DIAGNOSTIC_ONLY_input.mp4"
        writer = imageio_ffmpeg.write_frames(str(source), (96, 64), fps=2,
                    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", ffmpeg_log_level="error")
        writer.send(None)
        try:
            for value in (32, 96, 160, 224):
                writer.send(np.full((64, 96, 3), value, dtype=np.uint8).tobytes())
        finally:
            writer.close()
        report["diagnostic_media_io"] = render_video(model, source, args.output / "DIAGNOSTIC_ONLY_video.mp4", seconds=2)
        report["status"] = "PASS_DIAGNOSTIC_ONLY"
        if not all(value["passed"] for value in report["training_sizes"].values()):
            report["status"] = "PARTIAL_SIZE_OOM_REQUIRES_CONFIG_REVIEW"
        report["real_content_training_ready"] = False
        report["remaining"] = "Upload, audit/split and real-content short-run gate before formal training"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["warnings"] = sorted({str(w.message) for w in captured})
        write_json(args.output / "result.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
