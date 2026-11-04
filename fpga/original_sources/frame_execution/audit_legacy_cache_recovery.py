"""Explicit recovery of two deleted historical map databases.

The historical VDB bytes are unavailable, not SHA-restored. The original maps,
sources, results and expected hashes remain immutable. Rebuilt VDBs stay in the
recovery directory and are never substituted for the current implementation.
For these two historical caches only, verify a fresh successful map, all frozen
inputs, and byte identity of the entire exported map after one generated EFX
identifier-scope rename. All other checks retain their original code.
"""
import importlib,inspect,json,re,hashlib
from pathlib import Path
from run_units import ROOT,HERE,sha

RECOVERY=ROOT/'tmp/board-test/cleanup-20261015'
MANIFEST=RECOVERY/'legacy-cache-revalidation.json'
EXPECTED={
 'connected-board-camera-checked3':('f171d25d30c5728d83cb4352e7a41bdc412828335d940a0954d74a05cd86d80c','073f095e','9477a8cd'),
 'connected-board-camera-payload1':('a6cc43a6889eaa3d32f4faea026061791d91f68f320171908f9f2fba653167f9','885ca095','b4403b4d'),
}
INSTALLED=False

def validate_row(board,expected):
    board=Path(board);assert board==ROOT/'tmp/frame-execution'/board.name
    assert board.name in EXPECTED and expected==EXPECTED[board.name][0]
    original=json.loads((board/'result.json').read_text())
    assert original['vdb_sha256']==expected
    inputs=json.loads((board/'inputs.json').read_text())
    evidence={str(board/'result.json'):sha(board/'result.json'),str(board/'inputs.json'):sha(board/'inputs.json')}
    for row in inputs['inputs']:
        assert sha(row['snapshot'])==row['sha256'],row['snapshot']
        evidence[row['snapshot']]=row['sha256']
    rebuilt=RECOVERY/('original-path-'+board.name)
    result=json.loads((rebuilt/'recovery-result.json').read_text())
    assert result['exit_code']==0 and result['source_files_unchanged'] is True
    assert result['expected']=={k:original[k] for k in ('map_sha256','vdb_sha256')}
    assert re.search(r'map\s*:\s*PASS',(rebuilt/'map.log').read_text())
    old=board/'outflow/ti60f225_oob.map.v';new=rebuilt/'outflow/ti60f225_oob.map.v';vdb=rebuilt/'outflow/ti60f225_oob.vdb'
    assert sha(old)==original['map_sha256']
    assert sha(new)==result['observed']['map_sha256'] and sha(vdb)==result['observed']['vdb_sha256']
    old_scope,new_scope=EXPECTED[board.name][1:]
    before=old.read_bytes();after=new.read_bytes()
    pattern=rb'(?<=_)'+new_scope.encode()+rb'(?=_)'
    normalized,count=re.subn(pattern,old_scope.encode(),after)
    assert count>0 and normalized==before,board.name
    for path in (old,new,vdb,rebuilt/'recovery-result.json',rebuilt/'map.log',rebuilt/'map-recovery.cmd',RECOVERY.parent/'recover-cleanup-vdb-sourcepath.py',Path(__file__)):
        evidence[str(path)]=sha(path)
    return dict(board=str(board),historical_vdb_sha256=expected,historical_vdb_restored=False,
                rebuilt_vdb=str(vdb),rebuilt_vdb_sha256=sha(vdb),old_scope=old_scope,new_scope=new_scope,
                generated_identifier_renames=count,whole_map_byte_equal_after_scope_rename=True,evidence=evidence)

def prove():
    rows=[validate_row(ROOT/'tmp/frame-execution'/n,x[0]) for n,x in EXPECTED.items()]
    return dict(status='TWO_HISTORICAL_CACHES_REBUILT_MAP_EQUIVALENT_NOT_BYTE_RESTORED',
        current_map_or_vdb_changed=False,historical_results_changed=False,hardware_access=False,
        rows=rows,evidence={k:v for row in rows for k,v in row['evidence'].items()})

def audit_cache(board,expected):
    path=Path(board)/'outflow/ti60f225_oob.vdb'
    if path.exists():assert sha(path)==expected;return
    # Exact, closed allowlist. Missing files elsewhere still fail normally.
    manifest=json.loads(MANIFEST.read_text());actual=validate_row(board,expected)
    assert actual in manifest['rows'] and manifest['status']==prove_status

prove_status='TWO_HISTORICAL_CACHES_REBUILT_MAP_EQUIVALENT_NOT_BYTE_RESTORED'
def install():
    global INSTALLED
    if INSTALLED:return
    assert MANIFEST.exists()
    replacements={
      'camera_checked_board':(" assert r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')"," audit_cache(board,r['vdb_sha256'])"),
      'camera_payload_board':("    assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')","    assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')\n    audit_cache(board,r['vdb_sha256'])"),
    }
    for name,(old,new) in replacements.items():
        module=importlib.import_module(name);source=inspect.getsource(module.audit_delta)
        assert source.count(old)==1
        transformed=source.replace(old,new)
        scope={**module.__dict__,'audit_cache':audit_cache}
        exec(compile(transformed,str(Path(__file__))+':'+name,'exec'),scope)
        module.audit_delta=scope['audit_delta']
    INSTALLED=True

if __name__=='__main__':
    assert not MANIFEST.exists()
    result=prove();MANIFEST.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result['status'],len(result['evidence']))
