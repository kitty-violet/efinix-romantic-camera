"""Native 720p WCT2 reconstruction and game-reference ablation."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from .runtime import configure_workspace, file_hash, write_json
from .wavelet_game_teacher import WaveletTeacher, moments, matrix_power_spd, transform, REVISION


def tensor(image):
    return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].cuda().float() / 255


def picture(value):
    return Image.fromarray((value[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))


def verify_math(teacher):
    spec = importlib.util.spec_from_file_location("wct2_pinned_core", teacher.directory / "utils/core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(20261003)
    c, s = torch.rand(1, 32, 23, 31, device="cuda"), torch.rand(1, 32, 19, 27, device="cuda")
    mean, cov = moments(s)
    stats = {"mean": mean, "color": matrix_power_spd(cov, .5)}
    actual = transform(c, stats, .7)
    expected = module.feature_wct(c, s, alpha=.7, device="cuda")
    error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=3e-5, rtol=3e-5):
        raise AssertionError(f"WCT differs from author: {error}")
    if not torch.equal(c, transform(c, stats, 0)):
        raise AssertionError("Zero alpha changed features")
    return {"author_wct_gpu_max_abs": error, "zero_alpha_identity": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["quiet_stairs", "plaza_stone", "park_people"])
    ap.add_argument("--option", choices=["cat5", "sum"], default="cat5")
    args = ap.parse_args()
    root = configure_workspace()
    run = root / "runs/wavelet-game-probe-20261003"
    prior = root / "runs/styleid-material-probe-20261002/clean_reference720"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher = WaveletTeacher(root, option=args.option)
    math_check = verify_math(teacher)
    print("MATH", math_check, flush=True)
    garden_path = run / "references/garden_scene.webp"
    garden = Image.open(garden_path).convert("RGB").crop((0, 0, 2000, 1110))
    garden = ImageOps.fit(garden, (1280, 720), Image.Resampling.LANCZOS)
    references = {"garden": garden, "old_building": Image.open(prior / "quiet_stairs/reference.png").convert("RGB")}
    for name, image in references.items():
        image.save(run / "references" / f"{name}_720.png")
        teacher.prepare_style(name, tensor(image))
    records = []
    variants = [("reconstruction", None, (), 0),
                ("old_building_encoder", "old_building", ("encoder",), .7),
                ("garden_encoder", "garden", ("encoder",), .7),
                ("garden_decoder", "garden", ("decoder",), 1.),
                ("garden_encoder_skip", "garden", ("encoder", "skip"), .7)]
    if args.option == "sum":
        variants = [("sum_reconstruction", None, (), 0),
                    ("sum_garden_encoder_05", "garden", ("encoder",), .5),
                    ("sum_garden_encoder_08", "garden", ("encoder",), .8),
                    ("sum_garden_encoder_skip", "garden", ("encoder", "skip"), .65)]
    for case in args.cases:
        dst = run / case
        dst.mkdir(exist_ok=True)
        source = Image.open(prior / case / "source.png").convert("RGB")
        if source.size != (1280, 720):
            raise ValueError("Unexpected source dimensions")
        source.save(dst / "source.png")
        x = tensor(source)
        for name, style, stages, alpha in variants:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            y = teacher(x, style, stages, alpha)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            path = dst / f"{name}.png"
            picture(y).save(path)
            record = {"case": case, "variant": name, "style": style, "stages": stages, "alpha": alpha,
                      "size": list(source.size), "seconds": seconds,
                      "rgb_mae": float((y - x).abs().mean() * 255),
                      "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "output": str(path), "sha256": file_hash(path),
                      "source_sha256": file_hash(dst / "source.png")}
            records.append(record)
            print(json.dumps(record), flush=True)
            report_name = "report.json" if args.option == "cat5" else "report_sum.json"
            write_json(run / report_name, {"status": "STILL_TEACHER_PROBE_NOT_NEW_TRAINED_MODEL",
                "method": f"WCT2 {args.option}; full FP32; style covariance cached; no segmentation or face pasting",
                "revision": REVISION, "math_checks": math_check, "records": records,
                "code_sha256": {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("wavelet_game_teacher.py")]},
                "hardware_contract_sha256": file_hash(root.parent / "network_ab/model_b_contract.json"),
                "warning": "No temporal/generalization/hardware acceptance. MAE is not perceptual quality."})


if __name__ == "__main__":
    main()
