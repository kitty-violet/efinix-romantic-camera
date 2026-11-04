"""Read-only retained-graph cost and identical primitive survey.

Candidates are not edits or equivalence/release evidence. Counts deliberately
separate shared combinational cones from uniquely attributable hardware.
"""
import argparse, collections, json, re, time
from pathlib import Path
from run_units import ROOT, sha
from video_service_graph import load
from camera_joint_native import OUT, used_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    native, out = a.native.resolve(), a.output.resolve()
    assert out.is_relative_to(ROOT / 'tmp') and not out.exists()
    start = time.monotonic()
    proof = json.loads((native / 'native-graph-audit.json').read_text())
    assert proof['status'] == 'SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
    for f, h in proof['inputs'].items():
        assert sha(f) == h, f
    d = load()
    graft = json.loads((native / 'graft.json').read_text())
    removed = set(graft['removed'])
    changes = {}
    for row in graft['changed_loads']:
        changes.setdefault((row['cell'], row['pin']), {})[tuple(row['from_token'])] = tuple(row['to_token'])
    g = {n: (k, {p: [changes.get((n, p), {}).get(t, t) for t in ts] for p, ts in pins.items()})
         for n, (k, pins) in d['g'].items() if n not in removed}
    assert len(g) == proof['retained_primitives']
    driver = {}
    loads = collections.defaultdict(set)
    for n, (k, pins) in g.items():
        for pin, ts in pins.items():
            for t in ts:
                if pin in OUT[k]:
                    driver[t] = n
                else:
                    loads[t].add((n, pin))
    # State/hard-block ownership by retained physical hierarchy. LUTs may
    # have tool-generated names, so trace their combinational input cones.
    def category(n):
        for key, fragments in (
            ('cnn', ('/cnn/',)), ('display', ('/reader/', '/upscaler/', '/output_fifo/', 'hdmi_')),
            ('camera', ('camera_', 'mipi_', '/capture/')),
            ('ddr', ('ddr3',)), ('host', ('cpu/',)),
        ):
            if any(s in n for s in fragments):
                return key
        return 'other'
    cones = collections.defaultdict(set)
    state_counts = collections.defaultdict(collections.Counter)
    for n, (k, pins) in g.items():
        if k in ('EFX_LUT4', 'EFX_ADD'):
            continue
        label = category(n)
        state_counts[label][k] += 1
        pending = [driver[t] for ts in used_inputs(k, pins, d['ps'][n]).values() for t in ts if t in driver]
        seen = set()
        while pending:
            c = pending.pop()
            if c in seen or g[c][0] not in ('EFX_LUT4', 'EFX_ADD'):
                continue
            seen.add(c)
            ck, cp = g[c]
            pending.extend(driver[t] for ts in used_inputs(ck, cp, d['ps'][c]).values() for t in ts if t in driver)
        cones[label].update(seen)
    memberships = collections.defaultdict(list)
    for label, nodes in cones.items():
        for n in nodes:
            memberships[n].append(label)
    cost = {label: dict(sequential_and_hard=dict(state_counts[label]),
                       combinational_reachable=len(cones[label]),
                       unique_comb=sum(len(memberships[n]) == 1 for n in cones[label]))
            for label in sorted(state_counts)}
    groups = collections.defaultdict(list)
    for n, (k, pins) in g.items():
        if k != 'EFX_LUT4' or n.startswith('protected_inst/'):
            continue
        # Literal all-pin equality preserves unknown-input semantics; no
        # don't-care reasoning or input support trimming in this inventory.
        key = (tuple(sorted(d['ps'][n].items())),
               tuple((pin, tuple(ts)) for pin, ts in sorted(pins.items()) if pin != 'O'))
        groups[key].append(n)
    duplicates = [dict(cells=ns, fanouts=[len(loads[g[n][1]['O'][0]]) for n in ns])
                  for ns in groups.values() if len(ns) > 1]
    result = dict(status='READ_ONLY_STRUCTURE_SURVEY_NOT_OPTIMIZATION_PROOF',
                  native_sha256=sha(native/'native_source.v'), checker_sha256=sha(__file__),
                  retained_cells=len(g), categories=cost,
                  shared_comb=dict(collections.Counter('+'.join(sorted(v)) for v in memberships.values() if len(v)>1)),
                  identical_lut_groups=duplicates, potential_duplicate_count=sum(len(x['cells'])-1 for x in duplicates),
                  hardware_access=False, source_edited=False, seconds=time.monotonic()-start)
    out.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('identical_lut_groups',)}, indent=2))


if __name__ == '__main__':
    main()
