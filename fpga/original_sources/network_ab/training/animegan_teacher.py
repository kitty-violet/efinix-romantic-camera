"""Pinned author AnimeGANv3 landscape inference; experimental teacher only."""
from pathlib import Path
import json
import numpy as np
import torch  # Loads the existing CUDA/cuDNN DLLs before ONNX Runtime.
import onnxruntime as ort
from .runtime import file_hash

REVISION = '73ada3910dafa87b4cc972bcd80ebfebce9f3abe'
MODEL_SHA = '95ba7b219073fd5b12f569bc38056ffd3019cf4caf15b1feb9f73d1286c9f69d'


class AnimeTeacher:
    def __init__(self, root, profile=None, cpu=False):
        self.path = Path(root) / 'cache/animegan-probe' / REVISION / 'AnimeGANv3_Hayao_36.onnx'
        if file_hash(self.path) != MODEL_SHA:
            raise ValueError('Author model drift')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        if profile:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile)
        providers = ['CPUExecutionProvider'] if cpu else [
            ('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC',
              'cudnn_conv_use_max_workspace': '0', 'gpu_mem_limit': str(4 * 2**30)}), 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers)
        if not cpu and self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('GPU provider unavailable; no silent CPU fallback')
        self.input = self.session.get_inputs()[0].name

    def __call__(self, rgb):
        if rgb.dtype not in (np.uint8, np.float32) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected HWC RGB in 0..255, uint8 or float32')
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError('RGB outside 0..255')
        h, w = rgb.shape[:2]
        if min(h, w) < 256 or h % 8 or w % 8:
            raise ValueError('Use explicitly prepared dimensions divisible by eight')
        x = np.ascontiguousarray(rgb[None], dtype=np.float32) / 127.5 - 1
        y = self.session.run(None, {self.input: x})[0]
        if y.shape != x.shape or not np.isfinite(y).all():
            raise FloatingPointError('Invalid author output')
        return np.clip((y[0] + 1) * 127.5, 0, 255).astype(np.uint8)

    def finish_profile(self):
        path = Path(self.session.end_profiling())
        events = json.loads(path.read_text(encoding='utf-8'))
        providers = {}
        for e in events:
            p = e.get('args', {}).get('provider')
            if p:
                providers[p] = providers.get(p, 0) + 1
        if not providers.get('CUDAExecutionProvider'):
            raise RuntimeError('No actual CUDA kernel in execution profile')
        return {'path': str(path), 'kernel_events_by_provider': providers}
