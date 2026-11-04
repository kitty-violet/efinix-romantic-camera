"""Verify frozen native-source/VDB lineage without re-running synthesis."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from complete_native_logic_split import resource_counts,protected_blocks
from probe_native_open_ports import compare_boundary


def audit():
    first=ROOT/'tmp/frame-execution/full-native-split-map3'
    final=ROOT/'tmp/frame-execution/full-native-split-map4'
    parent=ROOT/'tmp/frame-execution/connected-board-startup-cause4'
    bindings={}
    for folder in (first,final):
        inputs=json.loads((folder/'inputs.json').read_text())
        for name,h in inputs['files'].items():
            assert sha(folder/name)==h,(folder,name);bindings[str(folder/name)]=h
        bindings[str(folder/'inputs.json')]=sha(folder/'inputs.json')
    f=json.loads((first/'inputs.json').read_text())
    assert f['parent_map_sha256']==sha(parent/'outflow/ti60f225_oob.map.v')
    assert f['transform_sha256']==sha(HERE/'native_logic_split.py')
    a=json.loads((first/'public_connectivity_audit.json').read_text())
    assert a['status']=='PUBLIC_PRIMITIVE_RENAME_EQUIVALENCE_PASS'
    assert a['source_sha256']==sha(first/'native_source.v') and a['map_sha256']==sha(first/'outflow/ti60f225_oob.map.v')
    assert a['checker_sha256']==sha(HERE/'audit_native_roundtrip.py')
    g=json.loads((final/'inputs.json').read_text());r=json.loads((final/'result.json').read_text())
    assert g['parent_map_sha256']==a['map_sha256'] and g['parent_audit_sha256']==sha(first/'public_connectivity_audit.json')
    assert r['inputs_sha256']==sha(final/'inputs.json') and r['public_audit_sha256']==sha(final/'public_connectivity_audit.json')
    for name,key in [('ti60f225_oob.map.v','map_sha256'),('ti60f225_oob.vdb','vdb_sha256')]:
        assert sha(final/'outflow'/name)==r[key]
    for source,changed in ((parent/'outflow/ti60f225_oob.map.v',first/'native_source.v'),
                           (first/'outflow/ti60f225_oob.map.v',final/'native_source.v')):
        x,y=protected_blocks(source.read_text()),protected_blocks(changed.read_text())
        assert x and x==y, 'Encrypted source bytes changed'
    old=resource_counts(parent/'outflow/ti60f225_oob.map.rpt')
    new=resource_counts(final/'outflow/ti60f225_oob.map.rpt')
    assert old['EFX_COMB4']==128 and new.get('EFX_COMB4',0)==0
    expected=dict(old);expected['EFX_COMB4']=0;expected['EFX_LUT4']+=256
    assert {k:v for k,v in expected.items() if v}=={k:v for k,v in new.items() if v}
    final_audit=json.loads((final/'public_connectivity_audit.json').read_text())
    boundary=compare_boundary((final/'native_source.v').read_text(),
        (final/'outflow/ti60f225_oob.map.v').read_text(),final_audit['renamed_nets'])
    for p in (parent/'inputs.json',parent/'outflow/ti60f225_oob.map.v',first/'public_connectivity_audit.json',
              final/'result.json',final/'public_connectivity_audit.json',final/'outflow/ti60f225_oob.vdb',
              final/'outflow/ti60f225_oob.map.v',Path(__file__),HERE/'native_logic_split.py',
              HERE/'audit_native_roundtrip.py',HERE/'probe_native_open_ports.py'):
        bindings[str(p)]=sha(p)
    return dict(status='NATIVE_REIMPORT_CHAIN_AND_ENCRYPTED_BYTES_BOUND',required_evidence=bindings,
                resource_delta={'EFX_COMB4':-128,'EFX_LUT4':256},other_primitive_counts_unchanged=True,
                protected_source_unchanged=True,final_protected_boundary=boundary,physical_pass=False,hardware_access=False)


if __name__=='__main__':print(audit()['status'])
