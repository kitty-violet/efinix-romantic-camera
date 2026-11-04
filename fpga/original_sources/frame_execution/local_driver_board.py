"""Independent complete video/A board copy bound to local-driver map proof."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_split_board import BASE,audit_delta as parent_audit
PARENT=ROOT/'tmp/frame-execution/connected-board-native-split1'
MAPPED=ROOT/'tmp/frame-execution/native-local-drivers-map2'
FOUR=ROOT/'tmp/frame-execution/native-local-drivers-fourstate1'

def audit_delta(board):
    parent_audit(PARENT)
    meta=json.loads((board/'inputs.json').read_text());mapped=json.loads((MAPPED/'result.json').read_text())
    assert mapped['status']=='FULL_NATIVE_LOCAL_DRIVERS_MAP_GRAPH_PASS'
    assert mapped['lexical_audit']['status']=='LEXICAL_PUBLIC_NET_BIJECTION_PASS'
    assert meta['local_driver_parent']==str(PARENT) and meta['local_driver_parent_sha256']==sha(PARENT/'inputs.json')
    assert meta['local_driver_map_sha256']==sha(MAPPED/'result.json') and meta['local_driver_packager_sha256']==sha(__file__)
    for row in meta['inputs']:
        path=Path(row['snapshot']);assert path.is_relative_to(board) and sha(path)==row['sha256']
        rel=path.relative_to(board);origin=(MAPPED if rel.as_posix()=='native_source.v' else PARENT)/rel
        assert sha(path)==sha(origin),rel
    for name,key in (('ti60f225_oob.map.v','map_sha256'),('ti60f225_oob.vdb','vdb_sha256')):
        assert sha(board/'outflow'/name)==mapped[key]
    mi=json.loads((MAPPED/'inputs.json').read_text());proof=Path(mi['transform_result'])
    assert sha(proof)==mi['transform_result_sha256']
    d=json.loads(proof.read_text());assert d['escaped_identifiers_and_bit_selects_distinguished']
    assert d['source_sha256']==sha(MAPPED/'native_source.v')
    for p,h in d['inputs'].items():assert sha(p)==h,p
    fr=json.loads((FOUR/'result.json').read_text());fi=json.loads((FOUR/'inputs.json').read_text())
    assert fr['status']=='ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS' and fr['checks']==75264
    assert fr['inputs_sha256']==sha(FOUR/'inputs.json') and fr['log_sha256']==sha(FOUR/'simulation.log')
    assert fi['map_sha256']==mapped['map_sha256'] and fi['map_result_sha256']==sha(MAPPED/'result.json')
    return dict(status='LOCAL_DRIVER_COMPLETE_BOARD_SOURCE_AND_FOURSTATE_BOUND',
        original_functional_rtl_unchanged=True,cycles_added=0,protected_phy_unchanged=True,
        map_sha256=mapped['map_sha256'],fourstate_sha256=sha(FOUR/'result.json'),
        hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp') and not board.exists();parent_audit(PARENT)
    meta=json.loads((PARENT/'inputs.json').read_text());board.mkdir()
    for row in meta['inputs']:
        old=Path(row['snapshot']);rel=old.relative_to(PARENT)
        src=MAPPED/'native_source.v' if rel.as_posix()=='native_source.v' else old
        dest=board/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        row.update(snapshot=str(dest),sha256=sha(dest))
    shutil.copytree(MAPPED/'outflow',board/'outflow')
    meta.update(local_driver_parent=str(PARENT),local_driver_parent_sha256=sha(PARENT/'inputs.json'),
        local_driver_map_sha256=sha(MAPPED/'result.json'),local_driver_packager_sha256=sha(__file__),hardware_access=False)
    (board/'inputs.json').write_text(json.dumps(meta,indent=2))
    m=json.loads((MAPPED/'result.json').read_text())
    (board/'result.json').write_text(json.dumps(dict(status='LOCAL_DRIVER_COMPLETE_BOARD_MAPPED_NOT_RELEASE',
        inputs_sha256=sha(board/'inputs.json'),resources=m['counts'],map_sha256=m['map_sha256'],vdb_sha256=m['vdb_sha256'],
        hardware_access=False,board_download_allowed=False),indent=2))
    proof=audit_delta(board);(board/'delta_audit.json').write_text(json.dumps(proof,indent=2));print(proof['status'])

if __name__=='__main__':main()
