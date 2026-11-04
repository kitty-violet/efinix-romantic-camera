"""Conservative first-person foreground-skin cue for *training loss only*.

It is intentionally weak supervision, not a hand detector or user identity rule.
"""
import torch


def likely_foreground_skin(rgb):
    """Return N1HW soft mask in 0..1 for RGB 0..1 inputs."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("NCHW RGB expected")
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    h, w = rgb.shape[-2:]
    y = torch.arange(h, device=rgb.device).reshape(1, 1, h, 1) / h
    x = torch.arange(w, device=rgb.device).reshape(1, 1, 1, w) / w
    color = ((r > .35) & (g > .20) & (b > .12) &
             (r-g > .075) & (g-b > .018) & (r > g*1.08))
    spatial = (x < .68) & (y > .24)
    return (color & spatial).float()
