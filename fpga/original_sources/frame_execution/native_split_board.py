"""Package an official native-map candidate with immutable original RTL evidence.

The synthesis design_file is native_source.v. Original RTL copies support
traceability and existing interface checks; they are not remapped here.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from startup_cause_board import audit_delta as audit_parent

BASE=ROOT/'tmp/frame-execution/connected-board-startup-cause4'
MAPPED=ROOT/'tmp/frame-execution/full-native-split-map4'


def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text())
    parent=json.loads((BASE/'inputs.json').read_text())
    mapped=json.loads((MAPPED/'result.json').read_text())
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['native_map_result_sha256']==sha(MAPPED/'result.json')
    assert meta['native_transform_sha256']==sha(__file__)
    expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}|{'native_source.v'}
    seen=set()
    for row in meta['inputs']:
        path=Path(row['snapshot']);assert path.is_relative_to(board) and sha(path)==row['sha256']
        rel=path.relative_to(board).as_posix();seen.add(rel)
        origin=MAPPED/rel if rel in ('ti60f225_oob.xml','native_source.v') else BASE/rel
        assert sha(path)==sha(origin),rel
    assert seen==expected
    for name,key in [('ti60f225_oob.map.v','map_sha256'),('ti60f225_oob.vdb','vdb_sha256')]:
        assert sha(board/'outflow'/name)==mapped[key]
    assert mapped['status']=='FULL_NATIVE_ZERO_COMB_FRESH_MAP_PUBLIC_STRUCTURE_PASS'
    assert mapped['inputs_sha256']==sha(MAPPED/'inputs.json')
    proof=ROOT/'tmp/frame-execution/native-logic-split-fourstate128'
    four=json.loads((proof/'result.json').read_text())
    assert four['status']=='LOGIC_SPLIT_128_FOURSTATE_65536_PASS'
    assert four['inputs_sha256']==sha(proof/'inputs.json') and four['log_sha256']==sha(proof/'simulation.log')
    assert json.loads((proof/'inputs.json').read_text())['manifest_sha256']==sha(MAPPED/'all_replacements.json')
    return dict(status='NATIVE_128_SPLIT_ONLY_ORIGINAL_RTL_AND_FOURSTATE_BOUND',
                functional_rtl_unchanged=True,protected_rtl_unchanged=True,clock_pin_sdc_unchanged=True,
                input_sha256=sha(board/'inputs.json'),map_sha256=mapped['map_sha256'],
                fourstate_sha256=sha(proof/'result.json'),hardware_access=False,board_release=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp') and not board.exists()
    audit_parent(BASE);meta=json.loads((BASE/'inputs.json').read_text());board.mkdir()
    for row in meta['inputs']:
        old=Path(row['snapshot']);rel=old.relative_to(BASE)
        assert sha(old)==row['sha256']
        source=MAPPED/rel if rel.as_posix()=='ti60f225_oob.xml' else old
        target=board/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        row.update(snapshot=str(target),sha256=sha(target))
    src=MAPPED/'native_source.v';dst=board/src.name;shutil.copy2(src,dst)
    meta['inputs'].append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
    shutil.copytree(MAPPED/'outflow',board/'outflow')
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),native_map_result_sha256=sha(MAPPED/'result.json'),
                native_transform_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (board/'inputs.json').write_text(json.dumps(meta,indent=2))
    mapped=json.loads((MAPPED/'result.json').read_text());warnings=(board/'outflow/ti60f225_oob.warn.log').read_text()
    assert 'EFX-0654' not in warnings, 'combinational-loop warning'
    result=dict(status='CONNECTED_BOARD_NATIVE_MAP_MEASURED_NOT_RELEASE',inputs_sha256=sha(board/'inputs.json'),
                map_sha256=mapped['map_sha256'],vdb_sha256=mapped['vdb_sha256'],resources=mapped['counts'],
                native_source_result_sha256=sha(MAPPED/'result.json'),physical_timing_pass=False,board_download_allowed=False)
    (board/'result.json').write_text(json.dumps(result,indent=2))
    (board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2));print('NATIVE_SPLIT_BOARD_PREPARED')


if __name__=='__main__':main()
