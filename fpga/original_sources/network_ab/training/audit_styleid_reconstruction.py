"""Separate VAE reconstruction, DDIM reconstruction and style-injection damage.

This is a diagnosis on existing reviewed 720p images, not a new stylization,
training run or face-restoration result. No generated details are added to crops.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import AutoencoderKL

from .runtime import configure_workspace, file_hash, write_json


def main():
    root = configure_workspace()
    out = root / "runs/game-appearance-audit-20261003"
    out.mkdir(exist_ok=True)
    prior = root / "runs/styleid-material-probe-20261002/clean_reference720"
    info = json.loads((root / "runs/diffusion-teacher-preflight/sd15.json").read_text())
    vae_path = Path(info["snapshot"]) / "vae"
    expected = next(r["sha256"] for r in info["files"] if r["relative_path"] == "vae/diffusion_pytorch_model.fp16.safetensors")
    if file_hash(vae_path / "diffusion_pytorch_model.fp16.safetensors") != expected:
        raise ValueError("VAE changed")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = AutoencoderKL.from_pretrained(vae_path, variant="fp16", torch_dtype=torch.float32,
                                      use_safetensors=True, local_files_only=True).cuda().eval()
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 21)
    crops = {
        "plaza_stone": [("蓝衣人物脸部", (293, 470, 343, 530)), ("灰衣人物脸部", (778, 458, 828, 518))],
        "park_people": [("侧脸与衣领", (0, 235, 175, 505)), ("树丛与叶片", (875, 305, 1275, 605))],
        "quiet_stairs": [("台阶表面与接缝", (120, 400, 520, 700)), ("植被与邻接石栏", (690, 400, 1090, 700))],
    }
    report = []
    for case, regions in crops.items():
        folder = prior / case
        src = Image.open(folder / "source.png").convert("RGB")
        rgb = np.asarray(src).copy()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda().float() / 127.5 - 1
            z = vae.encode(x).latent_dist.mode()
            y = vae.decode(z).sample
            if not torch.isfinite(y).all():
                raise ValueError("Nonfinite VAE reconstruction")
            recon = ((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        recpath = out / f"{case}_vae_only.png"
        Image.fromarray(recon).save(recpath)
        files = [folder / "source.png", recpath, folder / "reconstruction.png", folder / "styleid_gamma_0.75.png"]
        labels = ["原始输入", "只经VAE编解码", "再经DDIM · 不加风格", "加游戏参考后的结果"]
        images = [Image.open(p).convert("RGB") for p in files]
        region_records = []
        for title, box in regions:
            board = Image.new("RGB", (1600, 365), "#141a20")
            draw = ImageDraw.Draw(board)
            diffs = []
            original = np.asarray(images[0].crop(box)).astype(float)
            for i, (label, im) in enumerate(zip(labels, images)):
                patch = im.crop(box)
                if patch.width < 100:
                    patch = patch.resize((patch.width * 5, patch.height * 5), Image.Resampling.NEAREST)
                    scale = "5倍最近邻查看 · 无补细节"
                elif patch.height > 300:
                    raise ValueError("Unexpected oversized ROI")
                else:
                    scale = "720p原尺度 · 未锐化"
                board.paste(patch, (i * 400 + (400 - patch.width) // 2, 65))
                draw.text((i * 400 + 6, 4), label, fill="white", font=font)
                draw.text((i * 400 + 6, 32), scale, fill="#c6d1dd", font=font)
                diffs.append(float(np.abs(np.asarray(im.crop(box)).astype(float) - original).mean()))
            name = f"{case}_region_{len(region_records)}.png"
            board.save(out / name)
            region_records.append({"label": title, "xyxy": box, "source_pixels_wh": [box[2]-box[0],box[3]-box[1]],
                "latent_span_wh_approx": [(box[2]-box[0])/8,(box[3]-box[1])/8],
                "rgb_mae_vs_source": dict(zip(labels, diffs)), "comparison": str(out / name)})
        report.append({"case": case, "files": [{"path": str(p), "sha256": file_hash(p)} for p in files],
            "regions": region_records, "latent_shape": list(z.shape),
            "vae_full_rgb_mae": float(np.abs(recon.astype(float)-rgb).mean()),
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20})
        print(json.dumps({"case": case, "vae_mae": report[-1]["vae_full_rgb_mae"], "regions":region_records},ensure_ascii=False),flush=True)
    write_json(out / "reconstruction_audit.json", {"status": "DIAGNOSTIC_NOT_NEW_MODEL_RESULT", "records":report,
        "vae_weights_sha256": expected, "code_sha256":file_hash(__file__),
        "warning": "ROI MAE measures pixel change, not identity or quality. No face recognition. Visual review required."})


if __name__ == "__main__":
    main()
