"""Keep 29 header/weight RAM anchors; free the 84 arena RAM locations.

The registered source has a changed connection topology. This is a measured
placement comparison, not permission to change RAM contents, ports or clocks.
DSP cascades retain their complete original assignments.
"""
import json
from pathlib import Path
from run_units import sha
import fixed_cnn_ram_locations as original
from fixed_control_ddr_reuse_locations import locations

ARENA=original.PREFIX+'core/arena/'


def generate(root,reference):
    root=Path(root)
    before=original.generate(root,reference)
    kept={n:p for n,p in before['assignments'].items() if not n.startswith(ARENA)}
    freed={n:p for n,p in before['assignments'].items() if n.startswith(ARENA)}
    assert len(kept)==29 and len(freed)==84
    path=root/'fixed_dsp.placeloc';all_before=locations(path)
    assert len(all_before)==401
    lines=path.read_text().splitlines()
    after='\n'.join(line for line in lines if not line.split() or line.split()[0] not in freed)+'\n'
    path.write_text(after,encoding='utf8')
    actual=locations(path)
    assert actual=={n:p for n,p in all_before.items() if n not in freed}
    assert len(actual)==317 and not set(actual).intersection(freed)
    record=dict(before,status='CNN_RAM_29_FIXED_84_ARENA_AUTOMATIC_PREPARED',
                assignments=kept,released_reference_assignments=freed,
                assignments_sha256=sha(path),generator_sha256=sha(__file__),
                parent_generator_sha256=sha(original.__file__),
                dsp_assignments_unchanged=True,ordinary_logic_fixed=False)
    (root/'cnn_ram_placement.json').write_text(json.dumps(record,indent=2))
    (root/'cnn_ram_generator.py').write_bytes(Path(__file__).read_bytes())
    return record


def verify_placed(root):
    root=Path(root);r=json.loads((root/'cnn_ram_placement.json').read_text())
    assert r['status']=='CNN_RAM_29_FIXED_84_ARENA_AUTOMATIC_PREPARED'
    assert r['generator_sha256']==sha(__file__)
    assert r['parent_generator_sha256']==sha(original.__file__)
    assert len(r['assignments'])==29 and len(r['released_reference_assignments'])==84
    base=Path(r['reference_physical']);canonical=Path(r['reference_canonical'])
    assert sha(canonical/'result.json')==r['reference_canonical_result_sha256']
    assert sha(base/'outflow/ti60f225_oob.out.placeloc')==r['reference_placement_sha256']
    assert sha(base/'outflow/ti60f225_oob.map.v')==r['reference_map_sha256']
    net=root/'outflow/ti60f225_oob.map.v'
    assert sha(net)==r['candidate_map_sha256']
    assert original.memory_cells(net)==original.memory_cells(base/'outflow/ti60f225_oob.map.v')
    assignment=root/'fixed_dsp.placeloc';assert sha(assignment)==r['assignments_sha256']
    anchors=locations(assignment);assert len(anchors)==317
    assert not set(anchors).intersection(r['released_reference_assignments'])
    place=root/'outflow/ti60f225_oob.out.placeloc';xy=locations(place)
    for n,p in anchors.items():assert xy[n]==p,(n,p,xy[n])
    names=set(r['assignments'])|set(r['released_reference_assignments'])
    assert len(names)==113 and all(n in xy for n in names)
    assert len({xy[n] for n in names})==113
    return dict(status='CNN_RAM_29_ANCHORS_AND_84_AUTOMATIC_ACTUAL_LOCATIONS_PASS',
                count=113,retained=29,released=84,
                locations={n:xy[n] for n in sorted(names)},
                released_moved=sum(list(xy[n])!=p for n,p in r['released_reference_assignments'].items()),
                placement_sha256=sha(place),policy_sha256=sha(root/'cnn_ram_placement.json'),
                auditor_sha256=sha(__file__),board_release=False)
