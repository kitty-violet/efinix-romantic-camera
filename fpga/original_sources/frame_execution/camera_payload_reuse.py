"""Bind unaffected camera tests through exact old/source/new mapped graphs."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from camera_payload_board import BOARD,PARENT,audit_delta
from native_local_drivers import graph
from probe_full_native_split_map import params

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();audit_delta(BOARD);root.mkdir()
    oldpath=PARENT/'outflow/ti60f225_oob.map.v';source=BOARD/'native_source.v';newpath=BOARD/'outflow/ti60f225_oob.map.v'
    oldtxt=oldpath.read_text();srctxt=source.read_text();newtxt=newpath.read_text()
    old,src,new=map(graph,(oldtxt,srctxt,newtxt));op,sp,np=map(params,(oldtxt,srctxt,newtxt))
    camera=json.loads((PARENT/'result.json').read_text())['new_cells']
    assert len(camera)>980
    from map_native_local_drivers import lexical_roundtrip
    lexical=lexical_roundtrip(srctxt,newtxt)
    for n in camera:
        assert old[n]==src[n] and op[n]==sp[n],('camera changed before map',n)
        assert src[n][0]==new[n][0] and sp[n]==np[n],('camera params',n)
    groups=json.loads((ROOT/'tmp/frame-execution/camera-payload-native2/groups.json').read_text())
    changed={x['cell'] for r in groups for x in r['loads']}
    assert not changed.intersection(camera)
    # The receiver/configuration observer cones do not consume modified RAM
    # payload nets. This avoids claiming inheritance across a changed input.
    changed_outputs={t for n in changed for t in src[n][1]['O']}
    assert not any(t in changed_outputs for n in camera for ts in src[n][1].values() for t in ts)
    bound={str(f):sha(f) for f in [oldpath,source,newpath,BOARD/'public_connectivity_audit.json',BOARD/'result.json',Path(__file__)]}
    for d in ('camera-checked-mapped-questa4','camera-observer-host2','camera-reader-offline1','camera-reader-budget1'):
        folder=ROOT/'tmp/board-test'/d;r=json.loads((folder/'result.json').read_text())
        bound[str(folder/'result.json')]=sha(folder/'result.json')
        if d=='camera-checked-mapped-questa4':
            assert r['status']=='CAMERA_ACTUAL_MAP_PIN_TESTS_PASS' and r['map_sha256']==sha(oldpath)
            assert {c['mode'] for c in r['cases'] if c['passed'] and not c['timed_out']}=={0,4,6,8}
            i=json.loads((folder/'inputs.json').read_text());assert r['inputs_sha256']==sha(folder/'inputs.json')
            bound[str(folder/'inputs.json')]=sha(folder/'inputs.json')
            for f,h in i['files'].items():bound[str(folder/f)]=h
            for c in r['cases']:bound[str(folder/f"mode{c['mode']}.log")]=c['log_sha256']
        else:bound.update(r['evidence'])
    for f,h in bound.items():assert sha(f)==h,f
    # Keep the snapshot bit layout accessible to the packaged reader, with
    # its original content and explicit old-map provenance.
    shutil.copy2(PARENT/'camera_graft.json',BOARD/'camera_graft.json')
    result=dict(status='UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_PAYLOAD_MAP',camera_cells=len(camera),
        parent_map_sha256=sha(oldpath),map_sha256=sha(newpath),lexical=lexical,
        camera_or_observer_pins_changed=0,modified_payload_is_not_camera_input=True,
        mapped_modes_reused=[0,4,6,8],evidence=bound,hardware_access=False,board_release=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],len(camera),flush=True)

if __name__=='__main__':main()
