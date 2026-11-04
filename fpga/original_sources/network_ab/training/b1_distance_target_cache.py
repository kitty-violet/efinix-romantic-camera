"""Lossless, content-addressed cache of training-only native SDF targets.

Cache the exact FP32 result on the requested device; do not change rounding,
resize a cached crop, or use target information during source inference.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .probe_b1_ordered_distance_geometry import distance_fields
from .build_b1_thirty_binary_regions import code_bits
from .runtime import file_hash
import inspect


class DistanceTargetCache:
    def __init__(self, root, compute=distance_fields):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.compute=compute
        self.hits=0;self.misses=0
        self.signature='|'.join(('native-sdf-fp32-v1',torch.__version__,
            file_hash(inspect.getsourcefile(compute)),file_hash(inspect.getsourcefile(code_bits))))

    def __call__(self, codes, gain=8):
        if tuple(codes.shape[-2:])!=(720,1280):return self.compute(codes,gain)
        array=codes.detach().cpu().numpy().astype('<u2',copy=False)
        header=f'{self.signature}|{codes.device}|{gain}|{tuple(codes.shape)}'
        h=hashlib.sha256(header.encode()+array.tobytes()).hexdigest();path=self.root/(h+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                assert str(z['key'])==h
                result=np.array(z['field'],copy=True)
            assert result.dtype==np.float32 and result.shape==(len(codes),30,180,320)
            self.hits+=1
            return torch.from_numpy(result).to(codes.device)
        field=self.compute(codes,gain)
        assert field.dtype==torch.float32
        np.savez_compressed(path,key=np.array(h),field=field.detach().cpu().numpy())
        self.misses+=1
        return field
