"""Production allocator/RTL contract for the B1 bank-local resource candidate.

Geometry only: does not alter network topology, quantization or tensor values.
Keep these checks on the producer side AND the rejection gates in the RTL.
"""
VERSION = 'B1_BANK_LOCAL_V1'


def check_fields(f):
    if f[0] != 2:
        assert f[0] in (0, 1) and f[6] in (0, 1), 'DMA opcode/format'
        alignment = 4 if f[6] else 16
        assert f[7] % alignment == 0, 'DMA address alignment'
        return None
    pw, stride = bool(f[6] & 1), bool(f[6] & 8)
    rgb = pw and f[28] == 7
    n = f[3] if pw else 2*f[3]+1 if stride else f[3]+2
    assert 1 <= n <= 6
    bases = f[11] | (f[12] << 32)
    seen = [[], [], []]
    mapping = []
    for row in range(n):
        plane = (f[10] >> (2*row)) & 3
        base = (bases >> (10*row)) & 1023
        assert plane < 3
        if base not in seen[plane]:
            seen[plane].append(base)
        assert len(seen[plane]) <= 2
        mapping.append(plane*2 + seen[plane].index(base))
    assert all(slot % 2 == 0 for slot in mapping[:3]), 'first-three row slots'
    assert n < 6 or mapping[5] not in (0, 4), 'sixth-row slot'
    assert not rgb or n <= 4
    return tuple(mapping)


def check_program(path):
    modes = {'compute': 0, 'dma': 0}
    mappings = set()
    for index, line in enumerate(path.read_text().splitlines()):
        word = int(line, 16)
        fields = [(word >> (32*i)) & 0xffffffff for i in range(32)]
        try:
            mapping = check_fields(fields)
        except AssertionError as exc:
            raise AssertionError(f'program[{index}]: {exc}') from exc
        modes['compute' if mapping is not None else 'dma'] += 1
        if mapping is not None:
            mappings.add(mapping)
    return dict(version=VERSION, commands=modes, row_maps=len(mappings))
