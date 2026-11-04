"""Bind the existing Gray constraints before native placement, not after it."""
import json
from pathlib import Path
from run_units import ROOT,sha
from control_ddr_clock_profile import apply_profile as original_profile
from rebind_native_placement import SOURCE,translate

REFERENCE=ROOT/'tmp/frame-execution/native-split-bound-place1'

def expected(root):
    record=json.loads((REFERENCE/'endpoint_rebinding.json').read_text())
    assert record['source']==str(SOURCE)
    for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb'):
        assert sha(root/name)==sha(SOURCE/name)==record['geometry'][name],name
    for path,h in record['original'].items():assert sha(path)==h,path
    before=(SOURCE/'ti60f225_oob.sdc').read_text()
    after,pins=translate(before,(SOURCE/'outflow/ti60f225_oob.map.v').read_text(),
                         (SOURCE/'outflow/ti60f225_oob.out.placeloc').read_text())
    assert pins==record['pins'] and len(pins)==140
    assert sum(a!=b for a,b in pins.items())==126
    return before,after,pins

def apply_profile(root,evidence):
    profile=original_profile(root,evidence)
    before,after,pins=expected(root)
    sdc=root/'ti60f225_oob.sdc'
    assert sdc.read_text()==before
    # This call runs before Interface Designer, pack and placement.
    assert not (root/'outflow/ti60f225_oob.place').exists()
    assert not (root/'prebound_constraints.json').exists()
    old_sha=sha(sdc);sdc.write_text(after,encoding='utf8')
    data=dict(status='EXACT_GRAY_NAMES_BOUND_BEFORE_PLACEMENT',
        source=str(SOURCE),reference=str(REFERENCE/'endpoint_rebinding.json'),
        reference_sha256=sha(REFERENCE/'endpoint_rebinding.json'),
        old_sdc_sha256=old_sha,new_sdc_sha256=sha(sdc),
        map_sha256=sha(root/'outflow/ti60f225_oob.map.v'),
        vdb_sha256=sha(root/'outflow/ti60f225_oob.vdb'),
        original={str(SOURCE/n):sha(SOURCE/n) for n in
                  ('ti60f225_oob.sdc','outflow/ti60f225_oob.out.placeloc')},
        pins=pins,buses=10,destination_bits=70,limit_ns=2.5,
        timing_values_unchanged=True,script_sha256=sha(__file__),
        hardware_access=False,board_release=False)
    (root/'prebound_constraints.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    profile.update(before_endpoint_rebinding_sha256=old_sha,after_sha256=sha(sdc),
        prebound_constraints_sha256=sha(root/'prebound_constraints.json'))
    return profile

def audit(root):
    root=Path(root);record=json.loads((root/'prebound_constraints.json').read_text())
    before,after,pins=expected(root)
    assert record['status']=='EXACT_GRAY_NAMES_BOUND_BEFORE_PLACEMENT'
    assert record['pins']==pins and (root/'ti60f225_oob.sdc').read_text()==after
    assert record['new_sdc_sha256']==sha(root/'ti60f225_oob.sdc')
    assert record['script_sha256']==sha(__file__)
    assert record['reference_sha256']==sha(record['reference'])
    for path,h in record['original'].items():assert sha(path)==h,path
    meta=json.loads((root/'physical_inputs.json').read_text())
    assert meta['clock_profile']['prebound_constraints_sha256']==sha(root/'prebound_constraints.json')
    actual=root/'outflow/ti60f225_oob.out.placeloc'
    if actual.exists():
        names={x.split()[0] for x in actual.read_text().splitlines() if x and not x.startswith('#')}
        assert all(pin.rsplit('|',1)[0] in names for pin in pins.values())
    return record
