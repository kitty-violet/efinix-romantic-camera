"""Resolve a verified immutable synthesis snapshot before applying RTL overrides."""
import json
from pathlib import Path
from run_units import ROOT, sha


def frozen_paths(paths, base):
    if base is None:
        return paths
    base = base.resolve()
    assert base.is_relative_to(ROOT/'tmp')
    manifest = json.loads((base/'inputs.json').read_text())
    result = json.loads((base/'result.json').read_text())
    assert sha(base/'inputs.json') == result['inputs_sha256']
    by_name = {}
    for record in manifest['inputs']:
        snapshot = Path(record['snapshot'])
        assert snapshot.is_relative_to(base) and sha(snapshot) == record['sha256']
        if snapshot.name in by_name:
            # The native DDR board keeps the same include at its root and
            # under assembly. Accept byte-identical copies only; never pick
            # an arbitrary same-named source with different contents.
            assert sha(by_name[snapshot.name]) == record['sha256'], snapshot.name
        else:
            by_name[snapshot.name] = snapshot
    return [by_name.get(path.name, path) for path in paths]
