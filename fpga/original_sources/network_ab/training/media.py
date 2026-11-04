"""One audited RGB decoder for phone review, training masters and inference.

Use the existing imageio-ffmpeg binary, not OpenCV's older bundled HEVC decoder.
HLG is converted to sRGB with a fixed 1000-nit reference / 100-nit SDR policy;
CUVA dynamic display metadata is not used. This is a reproducible conversion,
not a claim to reproduce the phone's proprietary display rendering.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path
import queue
import re
import subprocess
import threading

import imageio_ffmpeg
import numpy as np

from .runtime import file_hash

HLG_TO_SRGB = ("zscale=transfer=linear:npl=100,format=gbrpf32le,"
               "zscale=primaries=bt709,tonemap=tonemap=mobius:param=0.3:desat=2:peak=10,"
               "zscale=transfer=iec61966-2-1:matrix=gbr:range=full,format=rgb24")
FRAME = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+\deE.]+)")


@lru_cache(maxsize=1)
def decoder_identity():
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    version = subprocess.run([exe, "-version"], check=True, capture_output=True,
                             text=True, encoding="utf-8").stdout.splitlines()[0]
    return {"path": exe, "sha256": file_hash(exe), "version": version,
            "policy": "ffmpeg-rgb-srgb-v2-fixed-hlg-mobius"}


def probe_video(path):
    path = Path(path).resolve()
    return dict(_probe(str(path), path.stat().st_size, path.stat().st_mtime_ns))


@lru_cache(maxsize=64)
def _probe(path, size, modified):
    result = subprocess.run([decoder_identity()["path"], "-hide_banner", "-nostdin",
        "-threads", "2", "-copyts", "-i", path, "-map", "0:v:0", "-frames:v", "1",
        "-vf", "showinfo", "-f", "null", "-"], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode:
        raise RuntimeError(f"FFmpeg source probe failed: {Path(path).name}")
    lines = result.stderr.splitlines()
    frame = next((line for line in lines if "showinfo" in line and FRAME.search(line)), "")
    color = next((line for line in lines if "showinfo" in line and "color_range:" in line), "")
    shape = re.search(r"\bs:(\d+)x(\d+)", frame)
    rate = next((re.search(r"([\d.]+) fps", line) for line in lines
                 if "Video:" in line and " fps" in line), None)
    if not shape:
        raise ValueError(f"Missing decoded dimensions: {Path(path).name}")
    tags = dict(re.findall(r"(color_\w+):(\S+)", color))
    transfer = tags.get("color_trc", "unknown")
    if transfer == "arib-std-b67":
        if tags.get("color_primaries") != "bt2020":
            raise ValueError("HLG source primaries require a reviewed conversion")
        vf = HLG_TO_SRGB
    elif transfer in ("bt709", "smpte170m", "iec61966-2-1", "unknown", "unspecified"):
        vf = "format=rgb24"
    else:
        raise ValueError(f"Unreviewed video transfer function: {transfer}")
    return {"width": int(shape[1]), "height": int(shape[2]), "fps": float(rate[1]) if rate else None,
            "timestamp_origin_seconds": float(FRAME.search(frame)[3]),
            "color_tags": tags, "rgb_filter": vf, "autorotate": True,
            "decoder": decoder_identity(),
            "color_policy": "fixed HLG to SDR sRGB, 1000/100 nit assumption; no CUVA dynamic mapping"
                            if transfer == "arib-std-b67" else "SDR YUV to RGB; preserve source transfer"}


def decode_frames(path, *, indices=None, seconds=None, raw_color=False,
                  hlg_exposure_stops=0.0):
    """Yield (source frame index, relative decoded PTS seconds, RGB uint8 array).

    Selected indices refer to decoded display order, never index/average-fps.
    PTS is relative to the first displayed frame, including for selected reads;
    the original first PTS remains in probe metadata. Some phone clips start
    at a nonzero edit-list timestamp while OpenCV reports their first frame as 0.
    All modes disable FFmpeg's implicit duplicate/drop synchronization.
    ``raw_color`` is only for controlled diagnosis; production uses sRGB.
    """
    metadata = probe_video(path)
    if raw_color and hlg_exposure_stops:
        raise ValueError("Raw RGB and HLG exposure adjustment are incompatible")
    if hlg_exposure_stops and metadata["color_tags"].get("color_trc") != "arib-std-b67":
        raise ValueError("Exposure adjustment is only reviewed for HLG sources")
    if not -3.0 <= hlg_exposure_stops <= 0.0:
        raise ValueError("HLG exposure must be between -3 and 0 stops")
    selected = None if indices is None else list(indices)
    if selected is not None and (not selected or selected != sorted(set(selected)) or selected[0] < 0):
        raise ValueError("Selected frame indices must be nonempty, unique and sorted")
    filters = []
    if seconds is not None:
        if seconds <= 0:
            raise ValueError("Positive decode duration required")
        filters.append(f"trim=end={metadata['timestamp_origin_seconds']+seconds}")
    if selected is not None:
        filters.append("select='" + "+".join(f"eq(n,{n})" for n in selected) + "'")
    rgb_filter = "format=rgb24" if raw_color else metadata["rgb_filter"]
    if hlg_exposure_stops:
        rgb_filter = rgb_filter.replace("tonemap=tonemap=mobius",
            f"exposure=exposure={hlg_exposure_stops:g},tonemap=tonemap=mobius")
    filters.extend([rgb_filter, "showinfo"])
    cmd = [decoder_identity()["path"], "-hide_banner", "-nostdin", "-nostats",
           "-threads", "2", "-copyts", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-filter_threads", "2", "-vf", ",".join(filters), "-fps_mode", "passthrough"]
    if selected is not None:
        cmd.extend(["-frames:v", str(len(selected))])
    cmd.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps, errors = queue.Queue(), deque(maxlen=8)
    def drain():
        try:
            for data in process.stderr:
                line = data.decode("utf-8", errors="replace")
                match = FRAME.search(line) if "showinfo" in line else None
                if match:
                    timestamps.put((int(match[1]), float(match[3])))
                elif any(x in line.lower() for x in ("error", "invalid", "failed")):
                    errors.append(line.strip())
        finally:
            timestamps.put(None)
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    length = metadata["width"] * metadata["height"] * 3
    count, previous = 0, float("-inf")
    try:
        while True:
            data = process.stdout.read(length)
            if not data:
                break
            if len(data) != length:
                raise RuntimeError("Truncated decoded RGB frame")
            stamp = timestamps.get(timeout=30)
            if stamp is None or stamp[0] != count or stamp[1] < previous:
                raise RuntimeError("Decoder frame/PTS correspondence failed")
            index = count if selected is None else selected[count]
            previous = stamp[1]
            count += 1
            yield index, stamp[1]-metadata["timestamp_origin_seconds"], np.frombuffer(data, dtype=np.uint8).reshape(metadata["height"], metadata["width"], 3)
        code = process.wait(timeout=30)
        if code or count == 0 or (selected is not None and count != len(selected)):
            raise RuntimeError(f"Incomplete video decode ({count} frames, exit {code}): {'; '.join(errors)}")
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=30)
        thread.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
