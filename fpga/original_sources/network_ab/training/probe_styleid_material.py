"""Reference-conditioned still-image teacher probe; no model training/video claim."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from .runtime import configure_workspace, file_hash, write_json
from .styleid_attention import InjectionState, install_processors


class Teacher:
    def __init__(self, root, steps, max_timestep=999):
        info = json.loads((root / "runs/diffusion-teacher-preflight/sd15.json").read_text())
        self.info = info
        for row in info["files"]:
            path = Path(info["snapshot"]) / row["relative_path"]
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"Cached model hash mismatch: {path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            info["snapshot"], variant="fp16", torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True,
            safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(steps)
        self.timesteps = self.scheduler.timesteps[self.scheduler.timesteps <= max_timestep]
        if len(self.timesteps) < 2:
            raise ValueError("Inversion needs at least two timesteps")
        self.steps = steps
        self.state = InjectionState()
        self.layers = install_processors(self.pipe.unet, self.state)
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.text_encoder.to("cuda")
        tokens = self.pipe.tokenizer([""], padding="max_length",
                                    max_length=self.pipe.tokenizer.model_max_length,
                                    return_tensors="pt")
        with torch.inference_mode():
            self.text = self.pipe.text_encoder(tokens.input_ids.cuda())[0]
        self.pipe.text_encoder.to("cpu")
        self.pipe.unet.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()

    def encode(self, image):
        self.pipe.unet.to("cpu")
        torch.cuda.empty_cache()
        self.pipe.vae.to("cuda")
        x = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].cuda().float() / 127.5 - 1
        z = self.pipe.vae.encode(x).latent_dist.mode() * self.pipe.vae.config.scaling_factor
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        self.pipe.unet.to("cuda")
        return z.half()

    def decode(self, z):
        self.pipe.unet.to("cpu")
        torch.cuda.empty_cache()
        self.pipe.vae.to("cuda")
        y = self.pipe.vae.decode(z.float() / self.pipe.vae.config.scaling_factor).sample
        if not torch.isfinite(y).all():
            raise FloatingPointError("Nonfinite decoded image")
        rgb = ((y[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        self.pipe.unet.to("cuda")
        return Image.fromarray(rgb)

    def invert(self, z, mode):
        self.state.mode = mode
        for i, t in enumerate(reversed(self.timesteps)):
            self.state.timestep = int(t)
            noise = self.pipe.unet(z, t.to("cuda"), encoder_hidden_states=self.text).sample
            current_t = max(0, int(t) - 1000 // self.steps)
            a = self.scheduler.alphas_cumprod[current_t].to(z.device)
            b = self.scheduler.alphas_cumprod[int(t)].to(z.device)
            z = (z - (1 - a).sqrt() * noise) * (b.sqrt() / a.sqrt()) + (1 - b).sqrt() * noise
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite inversion")
            if (i + 1) % 5 == 0:
                print(f"{mode} inversion {i+1}/{len(self.timesteps)}; RAM features {self.state.cache_bytes()/2**20:.0f} MiB", flush=True)
        self.state.mode = "off"
        return z

    def generate(self, z, gamma=None, temperature=1.5):
        self.state.mode = "off" if gamma is None else "inject"
        self.state.gamma, self.state.temperature = gamma, temperature
        for i, t in enumerate(self.timesteps):
            self.state.timestep = int(t)
            noise = self.pipe.unet(z, t.to("cuda"), encoder_hidden_states=self.text).sample
            z = self.scheduler.step(noise, t, z).prev_sample
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite denoising")
            if (i + 1) % 5 == 0:
                print(f"generate gamma={gamma} {i+1}/{len(self.timesteps)}", flush=True)
        self.state.mode = "off"
        return self.decode(z)


def make_board(items, destination):
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
    w, h = 640, 360
    board = Image.new("RGB", (w * 2, (h + 42) * ((len(items) + 1) // 2)), "#141a20")
    draw = ImageDraw.Draw(board)
    for i, (label, img) in enumerate(items):
        x, y = (i % 2) * w, (i // 2) * (h + 42)
        board.paste(ImageOps.contain(img, (w, h)), (x, y + 42))
        draw.text((x + 8, y + 8), label, font=font, fill="white")
    board.save(destination, quality=97)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.75, 0.3])
    ap.add_argument("--max-timestep", type=int, default=999)
    ap.add_argument("--without-init-adain", action="store_true")
    args = ap.parse_args()
    root = configure_workspace()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261002)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    contract = root.parent / "network_ab/model_b_contract.json"
    if file_hash(contract) != cfg["hardware_contract_sha256"]:
        raise ValueError("Hardware contract changed")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    height = round(args.width * 9 / 16 / 8) * 8
    teacher = Teacher(root, args.steps, args.max_timestep)
    records = []
    for case in cfg["cases"]:
        if args.cases and case["name"] not in args.cases:
            continue
        started = time.perf_counter()
        dst = args.output / case["name"]
        dst.mkdir(exist_ok=True)
        for key in ("content", "style"):
            if file_hash(case[key]["path"]) != case[key]["sha256"]:
                raise ValueError(f"Changed {key} source")
        source = Image.open(case["content"]["path"]).convert("RGB").resize((args.width, height), Image.Resampling.LANCZOS)
        style = Image.open(case["style"]["path"]).convert("RGB").crop(case["style"]["safe_roi_xyxy"])
        # Preserve aspect ratio: only reference center-cropped to matching 16:9.
        style = ImageOps.fit(style, (args.width, height), Image.Resampling.LANCZOS)
        source.save(dst / "source.png"); style.save(dst / "reference.png")
        print(f"CASE {case['name']} {source.size}", flush=True)
        teacher.state.content.clear(); teacher.state.style.clear()
        gc.collect(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            style_z = teacher.invert(teacher.encode(style), "style")
            content_z = teacher.invert(teacher.encode(source), "content")
            reconstruction = teacher.generate(content_z.clone())
            reconstruction.save(dst / "reconstruction.png")
            # Channel-wise latent AdaIN from the author's implementation.
            latent = ((content_z - content_z.mean((2, 3), keepdim=True)) /
                      (content_z.std((2, 3), keepdim=True) + 1e-4) *
                      style_z.std((2, 3), keepdim=True) + style_z.mean((2, 3), keepdim=True))
            if args.without_init_adain:
                latent = content_z.clone()
            items = [("原始实拍 · 720p输入", source), ("你的游戏参考 · 去除周边UI", style)]
            outputs = []
            for gamma in args.gammas:
                image = teacher.generate(latent.clone(), gamma)
                name = f"styleid_gamma_{gamma:.2f}.png"
                image.save(dst / name)
                image.resize((1920, 1080), Image.Resampling.LANCZOS).save(dst / name.replace(".png", "_1080.png"))
                mae = float(np.abs(np.asarray(image).astype(float) - np.asarray(source)).mean())
                outputs.append({"file": str(dst / name), "gamma": gamma, "rgb_mae_vs_source": mae, "sha256": file_hash(dst / name)})
                items.append((f"StyleID γ={gamma} · 单张试验", image))
                make_board(items, dst / "comparison.jpg")
        record = {"case": case, "size": [args.width, height], "steps": args.steps,
                  "seconds": time.perf_counter() - started, "outputs": outputs,
                  "feature_cache_ram_mib": teacher.state.cache_bytes() / 2**20,
                  "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "reconstruction_rgb_mae": float(np.abs(np.asarray(reconstruction).astype(float) - np.asarray(source)).mean()),
                  "comparison": str(dst / "comparison.jpg")}
        records.append(record)
        write_json(args.output / "report.json", {"status": "STILL_TEACHER_REVIEW_REQUIRED", "records": records,
            "model_revision": teacher.info["revision"], "inference_dtype": "fp16 UNet, fp32 VAE",
            "max_timestep": args.max_timestep, "timesteps": teacher.timesteps.tolist(),
            "initial_adain": not args.without_init_adain,
            "method": "StyleID author's diffusers SD1.5 option with CPU feature RAM and SDPA",
            "code_sha256": {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("styleid_attention.py")]},
            "hardware_contract_sha256": file_hash(contract), "trained_weights": False,
            "warning": "Still-image teacher probe only; no temporal/generalization/deployment acceptance."})
        print(json.dumps({k: v for k, v in record.items() if k != "case"}), flush=True)


if __name__ == "__main__":
    main()
