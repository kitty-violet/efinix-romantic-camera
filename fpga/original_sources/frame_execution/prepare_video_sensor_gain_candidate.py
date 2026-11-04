"""Offline-only two-register gain candidate from frozen69. No implementation/download.

Keep exposure and timing; SC431HAI manual p23 recommends reducing gain first.
Store the small ROM plus exact native INIT replacement records, not a duplicate build.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from prepare_video_service_color_exposure import read_rom, sha, ROOT

NATIVE = ROOT/'tmp/frame-execution/video-service-native69-input-isp-explicit/native_source.v'
ROM = ROOT/'tmp/frame-execution/video-service-native67-color-exposure/candidate_i2c_master_reg_rom.v'
MANUAL = ROOT/'output/toolchain/project_evidence_audit_2026-09-28/pdf_text/摄像头_摄像头SC431HAI_数据手册_V1.1_dfe1737d7ee2.txt'
NATIVE_SHA = 'd0cad77f3965599630d56b29ac3e7b34cbb134f7736143a8998b0daccd478a9e'
CHANGES = {0xa1: (0x3e08, 0x80, 0x00), 0xa2: (0x3e09, 0x2a, 0x20)}


def build(root):
    if root.exists() or not root.is_relative_to(ROOT/'tmp/frame-execution'):
        raise ValueError('New project evidence directory required')
    assert sha(NATIVE) == NATIVE_SHA
    text = NATIVE.read_text(encoding='utf-8')
    words, blocks = read_rom(text)
    rom = ROM.read_text(encoding='utf-8')
    matches = re.findall(r"ROM\[8'h([0-9a-f]+)\]\s*=\s*\{16'h([0-9a-f]+),8'h([0-9a-f]+),1'b([01])\}", rom, re.I)
    declared = {int(a, 16): int(reg, 16)<<9 | int(v, 16)<<1 | int(last) for a, reg, v, last in matches}
    assert len(declared) == 164 and all(declared[i] == words[i] for i in range(164))
    manual = MANUAL.read_text(encoding='utf-8')
    assert "8'h80 8'h2A 2.021" in manual and "8'h00 8'h20 1.000" in manual
    lo, hi, block, width, blob = next(b for b in blocks if b[3] == 20)
    updated = blob
    for address, (register, old, new) in CHANGES.items():
        assert words[address] == register<<9 | old<<1
        mask = ((1<<width)-1) << (address*width)
        updated = (updated & ~mask) | (((register<<9 | new<<1)&((1<<width)-1)) << (address*width))
        pattern = rf"(ROM\[8'h{address:02x}\]\s*=\s*\{{16'h{register:04x},8'h){old:02x}"
        rom, count = re.subn(pattern, lambda m: m[1]+f'{new:02x}', rom, flags=re.I)
        assert count == 1
    newblock = block
    edits = []
    for i in range(40):
        old = f'{(blob>>(i*256))&((1<<256)-1):0256b}'
        new = f'{(updated>>(i*256))&((1<<256)-1):0256b}'
        if old == new: continue
        name = 'INIT_'+format(i, 'X')
        pattern = '('+name+r"\s*=\s*256'b)"+old
        newblock, count = re.subn(pattern, lambda m: m[1]+new, newblock)
        assert count == 2  # attribute and executable defparam agree
        edits.append(dict(parameter=name, old=old, new=new, occurrences=count))
    candidate = text[:lo]+newblock+text[hi:]
    after, _ = read_rom(candidate)
    different = [i for i in range(164) if words[i] != after[i]]
    assert different == sorted(CHANGES)
    for i in range(164):
        expected = CHANGES[i][0]<<9 | CHANGES[i][2]<<1 if i in CHANGES else words[i]
        assert after[i] == expected
    # All unaffected native text and all 164 register addresses / last flags retained.
    assert candidate[:lo]+block+candidate[lo+len(newblock):] == text
    assert all((a>>9, a&1) == (b>>9, b&1) for a, b in zip(words, after))
    assert [after[i] for i in (0x73, 0x74, 0x75)] == [words[i] for i in (0x73, 0x74, 0x75)]
    root.mkdir()
    target = root/'candidate_i2c_master_reg_rom.v'
    target.write_text(rom, encoding='utf-8')
    result = dict(status='OFFLINE_GAIN_ROM_EXACT_DELTA_PASS_NOT_RELEASED',
        source_sha256=sha(NATIVE), candidate_native_sha256=hashlib.sha256(candidate.encode('utf-8')).hexdigest(),
        candidate_native_written=False, candidate_rom_sha256=sha(target),
        gain_before=2.021, gain_candidate=1.0, relative_sensor_signal_estimate=1/2.021,
        changed_addresses=different, changes={str(k):dict(register=r, old=o, new=n) for k,(r,o,n) in CHANGES.items()},
        native_init_patches=edits, compared_register_words=164,
        untouched_native_text_byte_equal=True, exposure_unchanged_half_lines=0x0bad,
        clocks_pins_topology_A_unchanged=True, hardware_access=False, board_release=False,
        pending=['new-source physical and bitstream binding', 'sensor response and clipping/colour under same scene',
                 'no claim of recovering already saturated frames', 'runtime LUT is not copied into this ROM'],
        evidence={str(p):sha(p) for p in (Path(__file__),NATIVE,ROM,MANUAL)})
    (root/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    assert sha(NATIVE) == NATIVE_SHA
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path)
    r = build(p.parse_args().root.resolve())
    print(json.dumps({k:r[k] for k in ('status','changed_addresses','compared_register_words','exposure_unchanged_half_lines')}))
