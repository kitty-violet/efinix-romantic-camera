"""Read-only address witnesses for the frozen whole-board CNN program.

The original program JSON is hash checked and both packed streams are rebuilt
byte-for-byte before its geometry is used. This is not a lifetime rescheduler,
an RTL equivalence proof, or permission to reduce a physical RAM depth.
"""
import argparse
import ast
import collections
import json
from pathlib import Path
from run_units import HERE, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    board, vectors = args.board.resolve(), args.vectors.resolve()
    manifest = json.loads((vectors / 'frame_manifest.json').read_text())
    used = ['program.json', 'program.hex', 'compute_commands.hex',
            'compute_dependencies.hex', 'dma_commands.hex', 'dma_dependencies.hex']
    for name in used:
        assert sha(vectors / name) == manifest['files'][name], name
    # Read the existing ABI constants without executing its command-line main.
    compiler = HERE / 'compile_production_descriptors.py'
    widths = {}
    for node in ast.parse(compiler.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ('COMPUTE_WIDTHS', 'DMA_WIDTHS'):
                widths[target.id] = ast.literal_eval(node.value)
    program = [int(word, 16) for word in (vectors / 'program.hex').read_text().split()]
    packed_hashes = {}
    for stream, size in [('compute', 64), ('dma', 16)]:
        indices = [int(word, 16) for word in (vectors / (stream + '_commands.hex')).read_text().split()]
        deps = [int(word, 16) for word in (vectors / (stream + '_dependencies.hex')).read_text().split()]
        assert len(indices) == len(deps)
        binary = bytearray()
        for index, dep in zip(indices, deps):
            fields = [(program[index] >> (32 * j)) & 0xffffffff for j in range(32)]
            values = [(dep, 14)] + ([(index, 15)] if stream == 'compute' else [])
            values += [(fields[j], width) for j, width in widths[stream.upper() + '_WIDTHS'].items()]
            word = offset = 0
            for value, width in values:
                assert 0 <= value < 1 << width
                word |= value << offset
                offset += width
            assert offset <= size * 8
            binary += word.to_bytes(size, 'little')
        packed = board / (stream + '.hex')
        actual = b''.join(int(word, 16).to_bytes(16, 'little') for word in packed.read_text().split())
        assert actual == binary, f'{stream} is not the measured board program'
        packed_hashes[packed.name] = sha(packed)
    commands = json.loads((vectors / 'program.json').read_text())
    assert len(commands) == len(program)
    banks = [dict(bank=i, required_depth=0, witness=None) for i in range(12)]
    modes = collections.Counter()
    for index, command in enumerate(commands):
        assert command['command'] == index
        kind = command['kind']
        modes[kind] += 1
        rows = command['outputs'] if kind == 'compute' else [command['row']] if kind == 'load' else []
        for row in rows:
            assert 0 <= row['plane'] < 3 and row['groups'] in (1, 2, 4)
            # Highest real x in each of the four physical column banks. The
            # last channel group is real for every physical output row.
            for column_bank in range(4):
                if column_bank >= row['width']:
                    continue
                x = column_bank + 4 * ((row['width'] - 1 - column_bank) // 4)
                address = row['base'] + (x // 4) * row['groups'] + row['groups'] - 1
                assert 0 <= address < 1024
                bank = row['plane'] * 4 + column_bank
                if address + 1 > banks[bank]['required_depth']:
                    banks[bank].update(required_depth=address + 1,
                        witness=dict(command=index, node=command.get('node'), kind=kind,
                                     local_x=x, last_group=row['groups'] - 1,
                                     address=address, row=row))
    result = dict(status='FROZEN_BOARD_PROGRAM_WRITE_EXTENT_AUDIT', commands=dict(modes),
        banks=banks, depth_512_direct_reduction_safe=all(b['required_depth'] <= 512 for b in banks),
        snapshot_inputs_sha256=sha(board / 'inputs.json'),
        descriptor_hex_sha256=packed_hashes,
        vector_inputs_sha256={name: sha(vectors / name) for name in used},
        abi_compiler_sha256=sha(compiler), audit_sha256=sha(Path(__file__)),
        note='Actual load and compute-output write addresses under the frozen descriptor geometry. '
             'These witnesses disprove an unchanged-program cut to 512 words. They do not '
             'prove 1024 is globally optimal; repacking lifetimes would require a new program '
             'and dependency/numerical/full-frame verification.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'commands', 'depth_512_direct_reduction_safe')}))
    print('BANK_REQUIRED_DEPTHS', [bank['required_depth'] for bank in banks])


if __name__ == '__main__':
    main()
