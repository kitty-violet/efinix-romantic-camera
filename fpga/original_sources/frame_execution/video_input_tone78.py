"""Offline camera-input curve78; no RTL or sensor changes."""
import hashlib
import json
import math
import zlib
from pathlib import Path

GAINS = (.80017, .514045, .98945)
KNOTS = ((0, 0), (32, 32), (64, 82), (96, 122), (128, 154),
         (160, 179), (192, 198), (224, 212), (255, 224))


def tone(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 255:
        raise ValueError('Finite input code 0..255 required')
    for (x0, y0), (x1, y1) in zip(KNOTS, KNOTS[1:]):
        if value <= x1:
            return y0 + (value-x0)*(y1-y0)/(x1-x0)
    raise AssertionError('Unreachable')


def table():
    # Compose before rounding: do not quantize through the previous table twice.
    return bytes(int(tone(255*(x*gain/255)**(1/3))+.5)
                 for gain in GAINS for x in range(256))


def describe():
    data = table()
    return dict(kind='CAMERA_INPUT_TONE78', status='OFFLINE_ONLY_NOT_APPLIED',
                base_gamma=3.0, black=0, gains=list(GAINS), swap_rb=True,
                tone_knots=[list(p) for p in KNOTS], bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), crc32=f'{zlib.crc32(data):08x}',
                max_rgb=[max(data[c*256:(c+1)*256]) for c in range(3)],
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sensor_unchanged=dict(exposure_half_lines=1344, analog_gain=2.503),
                bitstream_version=76, added_fpga_logic=False,
                expected_previous_crc32='1c445090',
                limits=['Not a new Gamma scalar: Gamma3 plus a common piecewise tone curve.',
                        'Nonlinear channel curves can affect saturation; visual validation required.',
                        'Does not restore sensor/pre-LUT clipped or absent details.'])


def validate(data, plan):
    if plan != describe() or data != table():
        raise ValueError('Table/plan do not match the exact curve78 generator')
    if len(data) != 768:
        raise ValueError('Exactly768bytes required')
    for c in range(3):
        channel = data[c*256:(c+1)*256]
        if channel[0] != 0 or channel[1] == 0 or max(channel) > 224:
            raise ValueError('Curve endpoint/dead-zone policy')
        if list(channel) != sorted(channel):
            raise ValueError('Nonmonotone tone curve')
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path('E:/fpga/tmp/board-test')) or root.exists():
        raise ValueError('New project board-test directory required')
    data, plan = table(), describe()
    validate(data, plan)
    root.mkdir()
    (root/'table.bin').write_bytes(data)
    (root/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan))
