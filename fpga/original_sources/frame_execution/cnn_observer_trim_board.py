"""Independent full-video/A map with an explicit reduced observer profile."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from audit_legacy_cache_recovery import install
install()
from cnn_minimal_board import BOARD as PARENT,audit_delta as parent_audit
from run_units import ROOT,HERE,sha,run
from cnn_observer_trim_native import transform
from native_local_drivers import graph
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
from complete_native_logic_split import resource_counts,protected_blocks
from audit_observer_lut_folding import normalize

BOARD=ROOT/'tmp/frame-execution/connected-board-cnn-observer-trim1'
TRANSFORM=ROOT/'tmp/frame-execution/cnn-observer-trim-native1'

def transform_check():
    r=json.loads((TRANSFORM/'result.json').read_text())
    assert r['status']=='RETIRED_OBSERVER_CONE_AND_CAPTURE_REMOVAL_GRAPH_PASS'
    assert r['source_sha256']==sha(TRANSFORM/'native_source.v')
    for f,h in r['inputs'].items():assert sha(f)==h,f
    return r

def audit_delta(board):
    parent_audit(PARENT);transform_check();board=Path(board)
    m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
    assert r['status']=='CNN_OBSERVER_TRIM_FULL_BOARD_MAP_PASS'
    assert r['inputs_sha256']==sha(board/'inputs.json')
    for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
    for f,h in r['evidence'].items():assert sha(f)==h,f
    assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
    return dict(status='CNN_OBSERVER_TRIM_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    parent_audit(PARENT);proof=transform_check();b=BOARD
    if not a.audit_only:
        assert not b.exists();b.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
        for row in m['inputs']:
            src=Path(row['snapshot']);dest=b/src.relative_to(PARENT);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(TRANSFORM/'native_source.v' if dest.name=='native_source.v' else src,dest)
            row.update(snapshot=str(dest),sha256=sha(dest))
        m.update(observer_parent_sha256=sha(PARENT/'inputs.json'),observer_transform_sha256=sha(TRANSFORM/'result.json'),observer_builder_sha256=sha(__file__),hardware_access=False)
        (b/'inputs.json').write_text(json.dumps(m,indent=2))
        for n in ('map.cmd','camera_graft.json','recovery_graft.json','capture_burst_graft.json','replications.json'):
            shutil.copy2(PARENT/n,b/n)
        if a.prepare_only:print('PREPARED',b);return
        body,seconds=run(['cmd.exe','/d','/c','map.cmd'],b,'map.log',480)
    else:
        m=json.loads((b/'inputs.json').read_text());body=(b/'map.log').read_text();seconds=None
    for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
    assert re.search(r'map\s*:\s*PASS',body)
    source=(b/'native_source.v').read_text();mapped=(b/'outflow/ti60f225_oob.map.v').read_text()
    canonical,folding=normalize(source,mapped,proof['serializer_luts_with_retired_inputs'])
    (b/'serializer_constant_folding.json').write_text(json.dumps(folding,indent=2))
    conn=verify(flattened_public_ports(canonical),flattened_public_ports(mapped));assert conn['added_protected']==0
    lexical=lexical_roundtrip(canonical,mapped);boundary=compare_boundary(source,mapped,conn['renamed_nets'])
    (b/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
    expected=resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt');counts=resource_counts(b/'outflow/ti60f225_oob.map.rpt')
    for k,v in proof['removed_kinds'].items():expected[k]-=v
    assert expected==counts,(expected,counts)
    assert protected_blocks(source)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
    assert counts['EFX_RAM10']+counts['EFX_DPRAM10']==235
    assert counts.get('EFX_DSP48',0)+counts.get('EFX_DSP24',0)//2==160
    assert '[EFX-0654' not in (b/'outflow/ti60f225_oob.warn.log').read_text()
    r=dict(status='CNN_OBSERVER_TRIM_FULL_BOARD_MAP_PASS',seconds=seconds,counts=counts,lexical_audit=lexical,protected_boundary=boundary,
        map_sha256=sha(b/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(b/'outflow/ti60f225_oob.vdb'),inputs_sha256=sha(b/'inputs.json'),
        diagnostic_profile=proof['diagnostic_profile'],retired_payload_ranges=proof['retired_payload_ranges'],
        audit_only_after_initial_strict_pin_rejection=a.audit_only,
        evidence={str(f):sha(f) for f in (Path(__file__),HERE/'audit_observer_lut_folding.py',TRANSFORM/'result.json',b/'public_connectivity_audit.json',b/'serializer_constant_folding.json',b/'initial_map_builder.py')},
        hardware_access=False,physical_pass=False,board_release=False)
    assert not (b/'result.json').exists();(b/'result.json').write_text(json.dumps(r,indent=2));print(r['status'],counts,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if BOARD.exists():(BOARD/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
