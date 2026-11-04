"""Prepare an isolated state-minimization experiment, never a board release.

Pixel timing, table contents and command encoding remain unchanged. Diagnostic
CRC/index are only authoritative with valid/stable flags, as in the host gate.
Keep the frozen 69 source untouched until actual mapping and tests exist.
"""
import argparse, json
from pathlib import Path
from run_units import ROOT, HERE, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir()
    source = ROOT/'tmp/frame-execution/video-service-native69-input-isp-explicit/video_input_lut.sv'
    s = source.read_text()
    edits = [
        ("reg[9:0] count=0;reg[31:0] crc=32'hffffffff,accepted_crc=DEFAULT_CRC;",
         "reg[9:0] count=0;reg[31:0] crc=~DEFAULT_CRC;\n wire[31:0] accepted_crc=~crc;"),
        ('reg[15:0] expected_low=0,rejected=0;reg low_seen=0;',
         'reg[15:0] rejected=0;reg low_seen=0;'),
        ('reg[7:0] read_index=0,requested_index=0;',
         'reg[7:0] requested_index=0;wire[7:0] read_index=requested_index;'),
        ('read_value<={qb,qg,qr};read_index<=requested_index;',
         'read_value<={qb,qg,qr};'),
        ('expected_low<=config_data[15:0];low_seen<=1;',
         'low_seen<=config_data[15:0]==~crc[15:0];'),
        ('{config_data[15:0],expected_low}==~crc',
         'config_data[15:0]==~crc[31:16]'),
        ('accepted_crc<=~crc;', ''),
    ]
    for old, new in edits:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    s = s.replace('// Camera input correction only.',
                  '// CRC is authoritative only when table_valid; read index/value only when no read is pending.\n// Camera input correction only.', 1)
    target = root/'video_input_lut.sv'
    target.write_text(s, encoding='utf8')
    (root/'edit.json').write_text(json.dumps(dict(
        status='ISOLATED_INPUT_STATE_MINIMIZATION_PREPARED_NOT_VALIDATED',
        parent=str(source), parent_sha256=sha(source), candidate_sha256=sha(target),
        removed_state_bits=56, pixel_path_unchanged=True,
        changes=['Reuse running CRC as accepted CRC while valid.',
                 'Compare low CRC when its command arrives; full 32-bit acceptance preserved.',
                 'Reuse requested index; host already waits for pending flags to clear.'],
        semantic_limits=['During an invalid upload CRC identity is provisional.',
                         'While a read is pending index/value are not a completed sample.'],
        hardware_access=False, board_release=False,
        evidence={str(Path(__file__)):sha(__file__)}), indent=2), encoding='utf8')
    print(target)


if __name__ == '__main__':
    main()
