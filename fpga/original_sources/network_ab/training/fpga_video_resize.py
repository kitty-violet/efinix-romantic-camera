"""Integer video scaling matching the frozen FPGA, RGB uint8 only."""
import numpy as np


def area_3to2(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 required')
    h, w, _ = rgb.shape
    if h % 3 or w % 3:
        raise ValueError('Source axes must be divisible by 3')
    a = rgb.astype(np.uint16)
    horizontal = np.empty((h, w//3*2, 3), np.uint16)
    horizontal[:, 0::2] = 2*a[:, 0::3]+a[:, 1::3]
    horizontal[:, 1::2] = a[:, 1::3]+2*a[:, 2::3]
    out = np.empty((h//3*2, w//3*2, 3), np.uint8)
    out[0::2] = ((2*horizontal[0::3]+horizontal[1::3]+4)//9).astype(np.uint8)
    out[1::2] = ((horizontal[1::3]+2*horizontal[2::3]+4)//9).astype(np.uint8)
    return out


def _up_axis(rgb, axis):
    n = rgb.shape[axis]
    if n % 2:
        raise ValueError('Source axes must be divisible by 2')
    # (dst+.5)*2/3-.5 == (4*dst-1)/6; clamp each source index.
    numerator = 4*np.arange(n//2*3)-1
    lo, weight = np.divmod(numerator, 6)
    a = np.take(rgb, np.clip(lo, 0, n-1), axis=axis).astype(np.uint16)
    b = np.take(rgb, np.clip(lo+1, 0, n-1), axis=axis).astype(np.uint16)
    shape = [1]*rgb.ndim; shape[axis] = len(weight)
    weight = weight.astype(np.uint16).reshape(shape)
    return ((a*(6-weight)+b*weight+3)//6).astype(np.uint8)


def bilinear_2to3(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 required')
    return _up_axis(_up_axis(rgb, 1), 0)


def verify():
    from hw.efinity.prototypes.video_scaler.fixed_ratio_resize import resize_plane
    rng = np.random.default_rng(924)
    checks = 0
    for h, w in [(6, 6), (9, 12), (12, 15)]:
        for _ in range(4):
            a = rng.integers(0, 256, (h,w,3), dtype=np.uint8)
            b = area_3to2(a)
            for y in range(b.shape[0]):
                for x in range(b.shape[1]):
                    sy, sx = 3*(y//2)+y%2, 3*(x//2)+x%2
                    wy, wx = (2,1) if y%2==0 else (1,2), (2,1) if x%2==0 else (1,2)
                    expected = [
                        (sum(int(a[sy+dy,sx+dx,c])*wy[dy]*wx[dx] for dy in range(2) for dx in range(2))+4)//9 for c in range(3)]
                    assert b[y,x].tolist() == expected
                    checks += 3
            u = bilinear_2to3(b)
            for c in range(3):
                ref = np.array(resize_plane(b[:,:,c].tolist(), w, h), np.uint8)
                assert np.array_equal(ref,u[:,:,c]); checks += h*w
    for v in [0,1,127,128,254,255]:
        a = np.full((1080,1920,3), v, np.uint8)
        assert np.all(bilinear_2to3(area_3to2(a)) == v)
    return {'status':'PASS', 'scalar_comparisons':checks, 'native_constant_frames':6,
            'down':'coverage numerator rounded once with (sum+4)//9',
            'up':'half-pixel replicated-border, horizontal then vertical half-up integer rounding'}
