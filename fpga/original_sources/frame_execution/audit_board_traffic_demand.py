"""Frozen-program DDR byte demand; deliberately not a measured frame time.

One capture, one processed frame and one screen scan are charged per period.
Repeated scans and dropped captures change this ratio and require the runtime
traffic counters. Refresh/turnaround/command overhead has no payload byte and
must not be represented as an extra guessed byte count.
"""
import argparse
import ast
import collections
import json
from pathlib import Path
from run_units import HERE, sha


def beat_bytes(address, length):
    assert address >= 0 and length > 0
    return ((address % 16 + length + 15) // 16) * 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    board, vectors = args.board.resolve(), args.vectors.resolve()
    assert not args.output.exists()
    # The extent audit already rebuilds both packed hardware streams from
    # these source commands; verify that binding again before using geometry.
    extent_path = board / 'arena_extent_audit.json'
    extent = json.loads(extent_path.read_text())
    assert extent['snapshot_inputs_sha256'] == sha(board / 'inputs.json')
    for name, digest in extent['vector_inputs_sha256'].items():
        assert sha(vectors / name) == digest, name
    for name, digest in extent['descriptor_hex_sha256'].items():
        assert sha(board / name) == digest, name
    compiler = HERE / 'compile_production_descriptors.py'
    assert sha(compiler) == extent['abi_compiler_sha256']
    widths = next(ast.literal_eval(n.value) for n in ast.parse(compiler.read_text()).body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'DMA_WIDTHS')
    commands = json.loads((vectors / 'program.json').read_text())
    indices = [int(x, 16) for x in (vectors / 'dma_commands.hex').read_text().split()]
    words = [int(x, 16) for x in (board / 'dma.hex').read_text().split()]
    assert len(indices) == len(words) == 10440
    groups = collections.defaultdict(lambda: dict(commands=0, logical_bytes=0, bus_bytes=0))
    for index, word in zip(indices, words):
        fields, offset = {}, 14
        for field, width in widths.items():
            fields[field] = (word >> offset) & ((1 << width) - 1)
            offset += width
        row = commands[index]
        assert (fields[0] == 1) == (row['kind'] == 'store')
        assert fields[2] == row['row']['width']
        logical = fields[2] * (3 if fields[6] else 8 * fields[3])
        assert logical == row['row']['width'] * row['channels']
        group = groups[row['group'] + '_' + row['kind']]
        group['commands'] += 1
        group['logical_bytes'] += logical
        group['bus_bytes'] += beat_bytes(fields[7], logical)
    # Check the current fixed video geometry against its actual frozen sources.
    top = board / 'rtl/ti60f225_oob_top.v'
    raw = board / 'assembly/raw_ddr_frame_bridge.sv'
    display = board / 'assembly/rgb720_ddr_reader.sv'
    inputs = json.loads((board / 'inputs.json').read_text())
    frozen = {Path(row['snapshot']).resolve(): row['sha256'] for row in inputs['inputs']}
    for path in (top, raw, display):
        assert sha(path) == frozen[path.resolve()], path
    assert '.WORDS_PER_FRAME(518400)' in top.read_text()
    assert 'FRAME_BEATS = WORDS_PER_FRAME / 4;' in raw.read_text()
    display_text = display.read_text()
    assert "remaining<=compare_mode ? 9'd120:9'd240;" in display_text
    assert 'line_y!=719' in display_text
    descriptors = sum(len((board / name).read_text().split()) * 16
                      for name in ('compute.hex', 'dma.hex'))
    assert descriptors == 729536
    ledger = dict(cnn_dma=sum(v['bus_bytes'] for v in groups.values()),
                  raw_capture_write=518400 * 4, raw_frame_read=518400 * 4,
                  rgb720_capture_write=1280 * 720 * 3,
                  display_rgb720_read=1280 * 720 * 3,
                  compute_and_dma_descriptors=descriptors)
    total = sum(ledger.values())
    result = dict(status='FROZEN_BOARD_STATIC_TRAFFIC_DEMAND',
                  cnn_dma_groups=dict(groups), per_period_bus_bytes=ledger,
                  total_bus_bytes=total, required_bus_bytes_per_second_at_30fps=total * 30,
                  ddr_payload_peak_bytes_per_second=1600000000,
                  fraction_of_ideal_ddr_peak=total * 30 / 1600000000,
                  inputs_sha256=sha(board / 'inputs.json'), extent_audit_sha256=sha(extent_path),
                  sources_sha256={str(p): sha(p) for p in (top, raw, display)},
                  audit_sha256=sha(Path(__file__)), measured_frame_ms=None,
                  note='Conditional one capture/read/compute/display per frame; runtime stalls, '
                       'refresh, turnaround and repeated/dropped frames require actual measurement. '
                       'Coefficient/CPU boot traffic is one-time and not added to steady frame bytes.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'per_period_bus_bytes',
        'total_bus_bytes', 'required_bus_bytes_per_second_at_30fps', 'fraction_of_ideal_ddr_peak')}))


if __name__ == '__main__':
    main()
