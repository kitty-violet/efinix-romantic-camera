"""Review every sampled negative route path against another legal placement.

Only geometry is compared, not a claim about route delay. References from a
different map use retained native cells only, so reused mapper LUT numbers
cannot masquerade as the same logic. All cells are comparable on the same map.
"""
import argparse, collections, json, pickle, re
from pathlib import Path
from run_units import sha


def locations(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, x, y, sub = line.rsplit(None, 3)
        result[name] = (int(x), int(y), int(sub))
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ('report', 'failed', 'candidate', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--retained-cache', type=Path)
    a = p.parse_args()
    assert not a.out.exists()
    loc_files = [r/'outflow/ti60f225_oob.out.placeloc' for r in (a.failed, a.candidate)]
    maps = [r/'outflow/ti60f225_oob.map.v' for r in (a.failed, a.candidate)]
    same_map = sha(maps[0]) == sha(maps[1])
    retained = None
    if not same_map:
        assert a.retained_cache
        key, src, mapped, sp, mp = pickle.loads(a.retained_cache.read_bytes())
        assert key[1] == sha(maps[0])
        retained = set(src)
    old, new = map(locations, loc_files)
    rows = []
    for block in re.split(r'(?=^Slack\s*:)', a.report.read_text(), flags=re.M)[1:]:
        slack = float(re.search(r'^Slack\s*:\s*([\d.+-]+)', block)[1])
        if slack >= 0:
            continue
        end = re.search(r'^Path End\s*:\s*(.*)', block, re.M)[1].strip()
        data = re.split(r'^Data Path\s*$', block, flags=re.M)[1].split('Capture Clock Path', 1)[0]
        cells = []
        for line in data.splitlines():
            m = re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)', line)
            if not m:
                continue
            n = m[1].strip().split('|')[0]
            if not cells or cells[-1] != n:
                cells.append(n)
        assert len(cells) >= 2, end
        pairs = list(zip(cells, cells[1:]))
        compared = [(u,v) for u,v in pairs if u in old and v in old and u in new and v in new
                    and (retained is None or u in retained and v in retained)]
        distance = lambda pos, u, v: abs(pos[u][0]-pos[v][0]) + abs(pos[u][1]-pos[v][1])
        wire = sum(float(x) for x in re.findall(r'^\s*net \(fo=\d+\)\s+([\d.]+)', data, re.M))
        delay = float(re.search(r'^Data Path Delay\s*:\s*([\d.]+)', block, re.M)[1])
        rows.append(dict(end=end, slack_ns=slack, wire_ns=wire, data_ns=delay,
                         net_fraction=wire/delay, path_edges=len(pairs), compared_edges=len(compared),
                         failed_distance=sum(distance(old,u,v) for u,v in compared),
                         candidate_distance=sum(distance(new,u,v) for u,v in compared)))
    assert len(rows) == 509
    groups = collections.defaultdict(list)
    for r in rows:
        groups[re.sub(r'\[\d+\]', '[]', r['end'])].append(r)
    families = [dict(end=k, count=len(v), worst_ns=min(x['slack_ns'] for x in v),
                     mean_wire_fraction=sum(x['net_fraction'] for x in v)/len(v),
                     compared_edges=sum(x['compared_edges'] for x in v),
                     failed_distance=sum(x['failed_distance'] for x in v),
                     candidate_distance=sum(x['candidate_distance'] for x in v)) for k,v in groups.items()]
    families.sort(key=lambda x:x['worst_ns'])
    out = dict(status='NEGATIVE_PATH_GEOMETRY_COMPARISON_NOT_TIMING_PROOF',same_map=same_map,
               families=families, rows=rows, hardware_access=False,
               evidence={str(f.resolve()):sha(f) for f in [a.report, *loc_files, *maps, Path(__file__)]})
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(dict(paths=len(rows),same_map=same_map,families=families[:8])))


if __name__ == '__main__':
    main()
