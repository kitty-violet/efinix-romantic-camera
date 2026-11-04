"""Fine-tune a full-resolution AdaIN decoder on the user's audited game crops."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F

from .adain_model import channel_moments, load_pretrained, _sha256
from .data import Sampler, audit_style, image_tensor, load_content_manifest
from .engine import set_reproducible
from .evaluate import render_video, save_comparison
from .runtime import HERE, configure_workspace, file_hash, write_json


FEATURE_INDICES = (3, 10, 17, 30)


def encode_multi(encoder, image):
    features = []
    for index, layer in enumerate(encoder):
        image = layer(image)
        if index in FEATURE_INDICES:
            features.append(image)
    return features


def mean_std_loss(actual, reference):
    am, ast = channel_moments(actual)
    rm, rst = channel_moments(reference)
    return F.mse_loss(am, rm) + F.mse_loss(ast, rst)


class FixedStyle(torch.nn.Module):
    def __init__(self, encoder, decoder, style_image):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        with torch.no_grad():
            self.style = self.encoder(style_image)

    def forward(self, normalized_content):
        content = ((128*normalized_content+128)/255).clamp(0, 1)
        feat = self.encoder(content)
        cm, cs = channel_moments(feat)
        sm, ss = channel_moments(self.style)
        stylized = self.decoder((feat-cm)/cs*ss+sm).clamp(0, 1)
        return ((stylized*255-128)/128).clamp(-1, 127/128)


def visual_review(run, step, encoder, decoder, style, validation, videos, config):
    directory = run / f"validation_{step:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    ids = config["eval_style_ids"]
    styles = {item: next(r for r in style if Path(r["path"]).name.endswith(item + ".png"))
              for item in ids}
    indices = (0, 8, 28) if step < 500 else range(len(validation))
    rows = []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for item, record in styles.items():
            model = FixedStyle(encoder, decoder,
                               ((128*image_tensor(record["path"], "cuda")+128)/255).unsqueeze(0))
            for index in indices:
                source = validation[index]
                image = image_tensor(source["path"], "cuda").unsqueeze(0)
                output = model(image)
                picture = directory / f"comparison_{item}_{index:03d}.jpg"
                save_comparison(image[0], output[0], picture)
                rows.append({"style_id": item, "style_sha256": record["sha256"],
                             "content_sha256": source["sha256"], "index": index,
                             "comparison": str(picture), "sha256": file_hash(picture)})
            if step >= 500 and item == ids[0] and videos:
                video = render_video(model, videos[0]["path"], directory / "comparison.mp4", seconds=8)
                write_json(directory / "video.json", video)
    write_json(directory / "review.json", {"step": step, "images": rows,
                                          "status": "VISUAL_REVIEW_REQUIRED"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "adain_game_config.json")
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = configure_workspace()
    if (root / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Run must be inside isolated workspace/runs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["schema"] != "game-style-adain-finetune-v1" or not 0 < args.stop_after <= config["maximum_steps"]:
        raise ValueError("Invalid experiment configuration")
    set_reproducible(config["seed"])
    content = load_content_manifest(config["content_manifest"])
    style = audit_style(Path(config["style_manifest"]).parent)
    if len(style) != 75:
        raise ValueError("Expected all 75 reviewed game crops")
    groups = {name: [r for r in style if Path(r["path"]).name.split("_")[-1].startswith(name)]
              for name in ("1-", "2-", "3-")}
    if [len(groups[key]) for key in groups] != [9, 21, 45]:
        raise ValueError("Original source group coverage changed")
    balanced = [r for group in groups.values() for r in (group*(45//len(group)) + group[:45%len(group)])]
    if Counter(Path(r["path"]).name.split("_")[-1][0] for r in balanced) != Counter({"1":45,"2":45,"3":45}):
        raise RuntimeError("Balanced source sampling failed")
    sampler = Sampler([r for r in content["images"] if r["split"] == "train"], balanced, config)
    validation = [r for r in content["images"] if r["split"] == "val"]
    videos = [v for v in content["videos"] if v["split"] == "val"]
    encoder, decoder = load_pretrained(root / "cache/pretrained/adain")
    encoder, decoder = encoder.cuda(), decoder.cuda()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config["learning_rate"])
    binding = {"config_sha256": file_hash(args.config), "content_manifest_sha256": file_hash(config["content_manifest"]),
               "style_manifest_sha256": file_hash(config["style_manifest"]),
               "trainer_sha256": file_hash(Path(__file__)),
               "model_sha256": file_hash(HERE / "adain_model.py"),
               "encoder_sha256": _sha256(root / "cache/pretrained/adain/vgg_normalised.pth"),
               "initial_decoder_sha256": _sha256(root / "cache/pretrained/adain/decoder.pth"),
               "torch_version": str(torch.__version__)}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["binding"] != binding or state["config"] != config:
            raise ValueError("Checkpoint data/source/config mismatch")
        decoder.load_state_dict(state["decoder"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
        if args.stop_after <= step:
            raise ValueError("Stop-after must exceed resumed step")
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        source = args.run / "source"
        source.mkdir()
        for path in (HERE / "adain_model.py", Path(__file__), args.config):
            shutil.copy2(path, source / path.name)
        step = 0
        write_json(args.run / "run.json", {"binding": binding, "config": config,
                                           "unique_style_count": 75,
                                           "balanced_source_counts": {k: len(v) for k,v in groups.items()},
                                           "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
                                           "quality_gate": "UNREVIEWED"})

    def checkpoint():
        if (file_hash(Path(__file__)) != binding["trainer_sha256"]
                or file_hash(HERE / "adain_model.py") != binding["model_sha256"]
                or file_hash(args.config) != binding["config_sha256"]):
            raise RuntimeError("Training source changed during run")
        path = args.run / f"checkpoint_{step:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"step": step, "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict(), "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "binding": binding, "config": config}, temporary)
        os.replace(temporary, path)
        return path

    log = args.run / f"metrics_from_{step:06d}_{time.time_ns()}.jsonl"
    status = "RUNNING"
    try:
        while step < args.stop_after:
            decoder.train()
            started = time.perf_counter()
            content_image, style_image, spec = sampler.next(step, "cuda")
            content_image = ((128*content_image+128)/255).clamp(0, 1)
            style_image = ((128*style_image+128)/255).clamp(0, 1)
            with torch.no_grad():
                content_features = encode_multi(encoder, content_image)
                style_features = encode_multi(encoder, style_image)
                cm, cs = channel_moments(content_features[-1])
                sm, ss = channel_moments(style_features[-1])
                target = (content_features[-1]-cm)/cs*ss+sm
            generated = decoder(target)
            features = encode_multi(encoder, generated)
            content_loss = F.mse_loss(features[-1], target)
            style_loss = sum(mean_std_loss(a, b) for a,b in zip(features, style_features))
            edge_loss = (F.l1_loss(generated[:, :, 1:, :]-generated[:, :, :-1, :],
                                   content_image[:, :, 1:, :]-content_image[:, :, :-1, :])
                         + F.l1_loss(generated[:, :, :, 1:]-generated[:, :, :, :-1],
                                     content_image[:, :, :, 1:]-content_image[:, :, :, :-1]))
            loss = (config["content_weight"]*content_loss + config["style_weight"]*style_loss
                    + config["edge_weight"]*edge_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {"step": step, "content": content_loss.item(), "style": style_loss.item(),
                   "edge": edge_loss.item(), "total": loss.item(), "gradient": norm.item(),
                   "seconds": time.perf_counter()-started, "style_file": spec["style"]["path"]}
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 25 == 0 or step == args.stop_after:
                write_json(args.run / "progress.json", {"status": "RUNNING", "completed_steps": step,
                                                        "loss": row["total"], "seconds_last_step": row["seconds"]})
            if step in config["validation_steps"]:
                visual_review(args.run, step, encoder, decoder, style, validation, videos, config)
            if step % 250 == 0 or step == args.stop_after:
                checkpoint()
        status = "STOP_AFTER_REACHED"
    finally:
        write_json(args.run / "status.json", {"status": status, "completed_steps": step,
                                              "quality_gate": "UNREVIEWED",
                                              "last_checkpoint": str(args.run / f"checkpoint_{step:06d}.pt")
                                              if (args.run / f"checkpoint_{step:06d}.pt").exists() else None})


if __name__ == "__main__":
    main()
