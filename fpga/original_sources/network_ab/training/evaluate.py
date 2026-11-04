"""Full 720p validation pictures, metrics and side-by-side video previews."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from .data import image_tensor
from .losses import color_moment_loss, color_moments, gram, rgb01
from .runtime import file_hash, write_json
from .media import decode_frames, probe_video


def to_u8(value):
    return (rgb01(value.detach()).clamp(0, 1)*255).round().byte().cpu().permute(1, 2, 0).numpy()


def save_comparison(content, output, path):
    image = np.concatenate([to_u8(content), to_u8(output)], axis=1)
    Image.fromarray(image).save(path)


@torch.no_grad()
def mean_style_targets(criterion, style):
    accum = None
    for record in style:
        current = {key: gram(value) for key, value in criterion.extract(image_tensor(record["path"], "cuda").unsqueeze(0)).items()}
        accum = current if accum is None else {k: accum[k]+current[k] for k in accum}
    return {key: value/len(style) for key, value in accum.items()}


@torch.no_grad()
def mean_color_targets(style):
    return torch.cat([color_moments(image_tensor(record["path"], "cuda").unsqueeze(0))
                      for record in style]).mean(0, keepdim=True)


@torch.no_grad()
def validate_images(model, criterion, style_targets, records, directory, style_weight,
                    color_target=None, color_weight=0.0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.eval()
    rows = []
    for index, record in enumerate(records):
        image = image_tensor(record["path"], "cuda").unsqueeze(0)
        if list(image.shape[-2:]) != [720, 1280]:
            raise ValueError("Fixed validation must be full 720p")
        output = model(image)
        repeated = model(image)
        if not torch.equal(output, repeated):
            raise RuntimeError("Repeated FP32 eval inference is not bit-identical")
        features = criterion.extract(image)
        lc, ls = criterion(output, (features["relu2_2"], style_targets))
        lcolor = color_moment_loss(output, color_target).item() if color_target is not None else 0.0
        baseline = torch.stack([F.mse_loss(gram(features[k]), style_targets[k]) for k in style_targets]).mean()
        def edge_energy(t):
            return ((t[:, :, 1:, :]-t[:, :, :-1, :]).abs().mean() +
                    (t[:, :, :, 1:]-t[:, :, :, :-1]).abs().mean()).item()
        picture = directory / f"comparison_{index:03d}.png"
        save_comparison(image[0], output[0], picture)
        color = rgb01(output)
        rows.append({"source": record, "content_loss": lc.item(), "style_loss": ls.item(),
                     "color_moment_loss": lcolor,
                     "input_style_loss": baseline.item(),
                     "saturation_fraction": ((output <= -1) | (output >= 127/128)).float().mean().item(),
                     "rgb_mean": color.mean((0, 2, 3)).tolist(),
                     "input_edge_energy": edge_energy(image), "output_edge_energy": edge_energy(output),
                     "repeat_inference_exact": True, "comparison": str(picture), "sha256": file_hash(picture)})
    if not rows:
        raise ValueError("No validation images")
    result = {"images": rows, "mean_content_loss": sum(r["content_loss"] for r in rows)/len(rows),
              "mean_style_loss": sum(r["style_loss"] for r in rows)/len(rows),
              "mean_color_moment_loss": sum(r["color_moment_loss"] for r in rows)/len(rows),
              "quality_gate": "REQUIRES_USER_REVIEW; NOT_QUANTIZATION_APPROVAL",
              "metric_limits": "Edge energy is diagnostic, not structural fidelity; LPIPS/flow audit pending"}
    result["validation_objective"] = (result["mean_content_loss"] + style_weight*result["mean_style_loss"]
                                      + color_weight*result["mean_color_moment_loss"])
    write_json(directory / "metrics.json", result)
    return result


@torch.no_grad()
def render_video(model, source, destination, *, seconds=8, hlg_exposure_stops=0.0):
    import imageio_ffmpeg
    metadata = probe_video(source)
    reader = decode_frames(source, seconds=seconds,
                           hlg_exposure_stops=hlg_exposure_stops)
    writer = None
    model.eval()
    count = 0
    try:
        fps = metadata["fps"]
        if fps is None or not 0 < fps <= 240:
            raise ValueError("Unsupported input video fps")
        writer = imageio_ffmpeg.write_frames(str(destination), (2560, 720), fps=fps,
                      codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                      macro_block_size=1, output_params=["-crf", "18", "-preset", "fast"], ffmpeg_log_level="error")
        writer.send(None)
        for index, timestamp, frame in reader:
            image = Image.fromarray(frame)
            image = ImageOps.fit(image, (1280, 720), method=Image.Resampling.BICUBIC)
            array = np.array(image, copy=True)
            value = (torch.from_numpy(array).permute(2, 0, 1).float().cuda()-128)/128
            output = model(value.unsqueeze(0))[0]
            writer.send(np.concatenate([array, to_u8(output)], axis=1).tobytes())
            count += 1
    finally:
        reader.close()
        if writer is not None:
            writer.close()
    if not count:
        raise RuntimeError("No video frames decoded")
    return {"frames": count, "fps": fps, "source_sha256": file_hash(source),
            "decode": metadata, "last_source_frame_index": index, "last_source_pts": timestamp,
            "timing": "all decoded frames retained; preview uses source reported average fps, not original VFR timing",
            "preview": str(destination), "sha256": file_hash(destination),
            "layout": "left=input; right=FP32; video has no audio", "seconds": count/fps,
            "hlg_exposure_stops": hlg_exposure_stops}
