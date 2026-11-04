"""Retained-map proof with one explicitly proved internal identity collapse."""
import itertools
import re
from pathlib import Path
from camera_joint_native import OUT
from native_local_drivers import lexical_tokens, CELL, REF
from audit_video_service_rq_fast import lut


def expand_cached_ports(g, text):
    ranges = {}
    for hi, lo, names in re.findall(r'^\s*wire\s+\[(\d+):(\d+)\]\s*([^;]+);', text, re.M):
        for name in re.sub(r'/\*.*?\*/', '', names, flags=re.S).split(','):
            name = name.strip().lstrip('\\')
            assert name and not any(c.isspace() for c in name)
            ranges[name] = int(hi), int(lo)
    for _, pins in g.values():
        for p, ts in pins.items():
            result = []
            for t in ts:
                if t[0] == 'net' and t[2] is None and t[1] in ranges:
                    hi, lo = ranges[t[1]]
                    step = -1 if hi > lo else 1
                    result.extend(('net', t[1], i) for i in range(hi, lo+step, step))
                else:
                    result.append(t)
            pins[p] = result


def prove_rq_identity(a, b, pa):
    name = 'LUT__140273'
    assert set(a)-set(b) == {name}, ('unexpected missing primitive', set(a)-set(b))
    kind, pins = a[name]
    assert kind == 'EFX_LUT4' and pa[name]['LUTMASK'] == (16, 0xf0f0)
    src, dst = pins['I2'][0], pins['O'][0]
    drivers = [(n, k, p) for n, (k, p) in a.items()
               if any(src in ts for port, ts in p.items() if port in OUT[k])]
    assert len(drivers) == 1 and drivers[0][1] == 'EFX_LUT4'
    n, _, p = drivers[0]
    # The actual upstream LUT emits 0/1/X, never Z, even for X/Z inputs.
    ins = [p.get('I'+str(i), [('literal', "1'b0", None)])[0] for i in range(4)]
    leaves = list(dict.fromkeys(t for t in ins if t[0] == 'net'))
    for values in itertools.product((0, 1, 'x', 'z'), repeat=len(leaves)):
        assert lut(pa[n]['LUTMASK'][1], ins, dict(zip(leaves, values))) in (0, 1, 'x')
    ins = [pins.get('I'+str(i), [('literal', "1'b0", None)])[0] for i in range(4)]
    assert {t for t in ins if t[0] == 'net'} == {src}
    for value in (0, 1, 'x'):
        assert lut(0xf0f0, ins, {src: value}) == value
    kept = 'LUT__133287'
    assert a[kept][0] == b[kept][0] == 'EFX_LUT4'
    kept_pins = a[kept][1]
    kept_ins = [kept_pins.get('I'+str(i), [('literal', "1'b0", None)])[0] for i in range(4)]
    assert {t for t in kept_ins if t[0] == 'net'} == {src}
    for value in (0, 1, 'x', 'z'):
        assert lut(pa[kept]['LUTMASK'][1], kept_ins, {src: value}) == lut(0xf0f0, ins, {src: value})
    replacement = kept_pins['O'][0]
    # The mapper merges the two equivalent buffers, retaining the real LUT.
    # Bind its actual output at every former load, including X/Z behavior.
    del a[name]
    for kind, ports in a.values():
        for port, ts in ports.items():
            if port not in OUT[kind]:
                ports[port] = [replacement if t == dst else t for t in ts]
    return dict(cell=name, input=src, output=dst, upstream=n,
                upstream_never_z=True, kept=kept, replacement=replacement,
                identity_values=['0', '1', 'X'], buffer_merge_values=['0', '1', 'X', 'Z'])


def bind_lut_boundaries(a, b, forward, bind, source):
    # Internal correspondence is established at named primitive OUTPUTS,
    # not LUT input positions: the mapper can permute inputs and LUTMASK.
    # Remaining named boundary signals must actually exist in the mapped
    # graph; the full truth table and protected-port audit below still apply.
    targets = {t for _, pins in b.values() for ts in pins.values() for t in ts}
    for n, (kind, pins) in a.items():
        if kind != 'EFX_LUT4':
            continue
        for p, ts in pins.items():
            if p == 'O':
                continue
            for t in ts:
                if t[0] == 'net' and t not in forward and t in targets:
                    bind(t, t, n+'.'+p+'.named_boundary')
    aliases = {}
    for left, right in re.findall(r'^\s*assign\s+([^;=]+)=([^;]+);', source, re.M):
        try:
            x = lexical_tokens(re.sub(r'/\*.*?\*/', '', left, flags=re.S))
            y = lexical_tokens(re.sub(r'/\*.*?\*/', '', right, flags=re.S))
        except AssertionError:
            continue  # An expression is not an identity and is not inferred.
        if len(x) == len(y) == 1:
            aliases[x[0]] = y[0]
    for _ in range(len(aliases)+1):
        changes = 0
        for x, y in aliases.items():
            if x in forward:
                continue
            target = forward.get(y, y if y in targets or y[0] == 'literal' else None)
            if target is not None:
                forward[x] = target
                changes += 1
        if not changes:
            break


def normalize_unused_outputs(a, b, pa, pb, source):
    zero = [('literal', "1'b0", None)]
    for g, ps in ((a, pa), (b, pb)):
        for n, (kind, pins) in g.items():
            if kind != 'EFX_LUT4':
                continue
            mask = ps[n]['LUTMASK'][1]
            for i in range(4):
                if all((mask >> v & 1) == (mask >> (v^(1<<i)) & 1) for v in range(16)):
                    pins['I'+str(i)] = zero
    loads = {t for kind, pins in a.values() for p, ts in pins.items() if p not in OUT[kind] for t in ts}
    exterior = CELL.sub('', source)
    exterior = re.sub(r'^\s*(?:defparam|wire)\b[^;]*;', '', exterior, flags=re.M)
    refs = {m['id'].lstrip('\\') for m in REF.finditer(exterior) if m['id']}
    dropped = []
    for n, (kind, pins) in a.items():
        if n not in b:
            continue
        for p in list(pins):
            if p not in OUT[kind] or p in b[n][1]:
                continue
            assert kind == 'EFX_ADD' and p in ('O', 'CO'), (n, p)
            assert all(t not in loads and (t[0] != 'net' or t[1] not in refs) for t in pins[p]), (n, p, pins[p])
            dropped.append(dict(cell=n, port=p, disconnected=pins[p]))
            del pins[p]
    return dropped


source = Path(__file__).with_name('audit_video_service_epoch_map.py')
code = source.read_text().replace("=='service_page_sequence[4]'", "=='service_legacy_epoch[1]'")
needle = ' # This alias is emitted explicitly'
insert = ' expand_cached_ports(a,source);expand_cached_ports(b,mapped)\n'+needle
assert code.count('code=source.read_text()') == 1
code = code.replace('code=source.read_text()', 'code=source.read_text().replace('+repr(needle)+','+repr(insert)+')')
old = 'del a[dead];discarded.append(dead)\n missing='
assert code.count(old) == 1
code = code.replace(old, 'del a[dead];discarded.append(dead)\n identity=prove_rq_identity(a,b,pa)\n unused_outputs=normalize_unused_outputs(a,b,pa,pb,source)\n missing=')
old = 'retained=len(a),discarded_unloaded_diagnostic_cells=discarded,added=len(added)'
assert code.count(old) == 1
code = code.replace(old, 'retained=len(a),discarded_unloaded_diagnostic_cells=discarded,collapsed_identity_luts=[identity],unused_adder_outputs=unused_outputs,added=len(added)')
needle = "exec(compile(code,str(source)+':unloaded-epoch-flag','exec'),globals())"
assert code.count(needle) == 1
updates = [
    ("   if n in folded and p!='O':continue", "   if kind=='EFX_LUT4' and p!='O':continue"),
    (' result=dict(status=', ' bind_lut_boundaries(a,b,forward,bind,source)\n result=dict(status='),
    ("('LUT_function',n)", "('LUT_function',n,q,z,pa[n],pb[n])"),
]
patches = ''.join('assert code.count('+repr(old)+')==1\ncode=code.replace('+repr(old)+','+repr(new)+')\n' for old,new in updates)
code = code.replace(needle, patches+needle)
exec(compile(code, str(source)+':rq-identity', 'exec'), globals())
