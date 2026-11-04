"""Shadow-neutral correction from the curve79 board sample; highlights retained."""
import hashlib
import json
import math
import zlib
from pathlib import Path
import video_input_tone79 as base

GAINS = base.GAINS
ANCHORS = (94.046875, 75.53515625, 104.0)
TARGET = 68.0
HIGHLIGHT = 160.0
SAMPLE_SHA = '2a8104c146ff4a503fd18f70f9b8ccf38389a8c0948b585afeb88aa0abfc811a'
BASE_SHA = '7a2f042280efad9f25d41fa10910166338ed0e9fc90c369794d4bd6981332d85'


def shadow(value, channel):
    if channel not in (0,1,2) or not math.isfinite(value) or not 0 <= value <= 255:
        raise ValueError('Finite RGB shadow input required')
    if value >= HIGHLIGHT:
        return value
    anchor = ANCHORS[channel]
    knots = ((0.,0.), (anchor/2,20.), (anchor,TARGET), (HIGHLIGHT,HIGHLIGHT))
    for (x0,y0),(x1,y1) in zip(knots,knots[1:]):
        if value <= x1:
            return y0+(value-x0)*(y1-y0)/(x1-x0)
    raise AssertionError('Unreachable')


def table():
    previous=base.table()
    if hashlib.sha256(previous).hexdigest()!=BASE_SHA:
        raise ValueError('The accepted highlight curve79 changed')
    data=[]
    for c,gain in enumerate(GAINS):
        for x in range(256):
            old=previous[c*256+x]
            # Copy every already-bright entry byte-for-byte. Compose darker
            # entries before their final rounding, not through an 8-bit inverse.
            y=base.tone(255*(x*gain/255)**(1/3))
            data.append(old if old>=HIGHLIGHT else int(shadow(y,c)+.5))
    return bytes(data)


def describe():
    data=table()
    return dict(kind='CAMERA_INPUT_TONE80',status='OFFLINE_ONLY_NOT_APPLIED',
        base_gamma=3.0,black=0,gains=list(GAINS),swap_rb=True,
        base_tone_knots=[list(k) for k in base.KNOTS],
        shadow_knots_rgb=[[[0,0],[a/2,20],[a,TARGET],[HIGHLIGHT,HIGHLIGHT]] for a in ANCHORS],
        unchanged_highlight_minimum_previous_code=160,
        bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=f'{zlib.crc32(data):08x}',
        max_rgb=[max(data[c*256:(c+1)*256]) for c in range(3)],
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_generator_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        base_table_sha256=BASE_SHA,sample_sha256=SAMPLE_SHA,
        sample_mean_rgb=list(ANCHORS),neutral_target=TARGET,
        sensor_unchanged=dict(exposure_half_lines=1344,analog_gain=2.503),
        bitstream_version=76,added_fpga_logic=False,expected_previous_crc32='88fe6505',
        limits=['Shadow correction uses one user-positioned neutral sample in current lighting; not a sensor calibration.',
                'Sample is post-ISP/area; table prediction is approximate, without a unique RAW inverse.',
                'Preserves previous LUT entries >=160, not every mixed-color pixel containing one bright channel.',
                'Cannot recover missing upstream detail; colored shadows may change saturation.'])


def validate(data,plan):
    if plan!=describe() or data!=table():
        raise ValueError('Exact curve80 table/plan required')
    previous=base.table()
    if len(data)!=768:
        raise ValueError('Exactly 768 bytes required')
    for c in range(3):
        a=data[c*256:(c+1)*256]
        if a[0]!=0 or any(v==0 for v in a[1:]) or list(a)!=sorted(a):
            raise ValueError('Monotonic nonclipping shadow table required')
    if any(b!=a for a,b in zip(previous,data) if a>=160) or any(b>a for a,b in zip(previous,data)):
        raise ValueError('Highlight preservation / shadow reduction failed')
    return data


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve()
    if not root.is_relative_to(Path('E:/fpga/tmp/board-test')) or root.exists():
        raise ValueError('New project board-test directory required')
    data,plan=table(),describe();validate(data,plan)
    root.mkdir();(root/'table.bin').write_bytes(data)
    (root/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(plan))
