"""Two uncut five-second clips, native 1080 -> 720 teacher -> 1080 output."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .runtime import configure_workspace, file_hash, write_json
from .wavelet_game_teacher import WaveletTeacher, REVISION
from .probe_wavelet_game import tensor, picture
from .render_temporal_native_detail import encode_pipe


def close_pipe(pipe):
    pipe.stdin.close()
    error = pipe.stderr.read().decode("utf-8", errors="replace")
    if pipe.wait(timeout=60):
        raise RuntimeError(error)


def video_check(path, expected):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    if fps != 30 or shape != [1920, 1080] or n != expected:
        raise RuntimeError(f"Video check failed: {path} {n} {fps} {shape}")
    return {"frames": n, "fps": fps, "size": shape, "seconds": n/fps,
            "bytes": path.stat().st_size, "sha256": file_hash(path)}


def main():
    root = configure_workspace()
    run = root / "runs/wavelet-game-probe-20261003"
    out = run / "video"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = WaveletTeacher(root)
    reference = run / "references/garden_720.png"
    net.prepare_style("garden", tensor(Image.open(reference)))
    manifest = root / "runs/coherent-surface-20261002/evaluation_clips.json"
    data = json.loads(manifest.read_text())
    chosen = [r for r in data["clips"] if r["id"] in ("holdout_corridor", "validation_plaza")]
    if len(chosen) != 2:
        raise ValueError("Missing frozen sources")
    combined = out / "wavelet_game_1080p30_10s.mp4"
    source_combined = out / "source_1080p30_10s.mp4"
    full_pipe = encode_pipe(combined, 1920, 1080, 30)
    source_pipe = encode_pipe(source_combined, 1920, 1080, 30)
    records = []
    for row in chosen:
        if file_hash(row["clip"]) != row["clip_sha256"]:
            raise ValueError("Source clip changed")
        cap = cv2.VideoCapture(row["clip"])
        target = out / (row["id"] + "_styled_1080p_5s.mp4")
        pipe = encode_pipe(target, 1920, 1080, 30)
        n, times, previous, duplicates = 0, [], None, 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if source.shape != (1080, 1920, 3):
                raise ValueError("Source is not native 1080p")
            small = Image.fromarray(source).resize((1280, 720), Image.Resampling.BICUBIC)
            x = tensor(small)
            torch.cuda.synchronize()
            start = time.perf_counter()
            y = net(x, "garden", ("encoder",), .7)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            output = picture(y)
            high = np.asarray(output.resize((1920, 1080), Image.Resampling.LANCZOS))
            raw = high.tobytes()
            if previous == raw:
                duplicates += 1
            previous = raw
            pipe.stdin.write(raw)
            full_pipe.stdin.write(raw)
            source_pipe.stdin.write(source.tobytes())
            if n in (0, 75, 149):
                output.save(out / f"{row['id']}_{n:03d}_native720.png")
                Image.fromarray(high).save(out / f"{row['id']}_{n:03d}_preencode1080.png")
            n += 1
            if n % 30 == 0:
                print(row["id"], n, "frames", "mean inference seconds", np.mean(times), flush=True)
        cap.release()
        close_pipe(pipe)
        if n != 150:
            raise RuntimeError("Not exactly five seconds")
        records.append({"source": row, "output": str(target), "validation": video_check(target, 150),
                        "mean_inference_seconds": float(np.mean(times)), "repeated_output_frames": duplicates})
        write_json(out / "progress.json", {"status": "RUNNING", "records": records})
    close_pipe(full_pipe)
    close_pipe(source_pipe)
    write_json(out / "report.json", {"status": "RENDERED_REVIEW_REQUIRED", "records": records,
        "output": str(combined), "output_validation": video_check(combined, 300),
        "source_output": str(source_combined), "source_validation": video_check(source_combined, 300),
        "reference_sha256": file_hash(reference), "revision": REVISION,
        "settings": {"option": "cat5", "stages": ["encoder"], "alpha": .7, "dtype": "float32",
                     "input": [1920,1080], "cnn": [1280,720], "output": [1920,1080],
                     "temporal_filter": False, "face_paste": False, "per_scene_tuning": False,
                     "encoder": "H264 CRF16 yuv420p, full frames, no optical-flow interpolation"},
        "code_sha256": {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("wavelet_game_teacher.py")]},
        "warning": "Offline reference-conditioned pretrained teacher, not new trained FPGA weights or realtime."})


if __name__ == "__main__":
    main()
