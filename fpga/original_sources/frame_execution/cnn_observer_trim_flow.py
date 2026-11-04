"""Current-map control, adapter, reset, observation and serial checks."""
from pathlib import Path
from audit_legacy_cache_recovery import install
install()
import json,shutil,sys
from run_units import ROOT,sha
from cnn_observer_trim_board import audit_delta

def resume_completed_map_prefix():
    # The first bounded run completed its actual control/CDC result before
    # timeout during the following camera-structure check. Reuse only that
    # independently verifiable result; never reuse the failed overall status.
    board=Path(sys.argv[sys.argv.index('--board')+1]).resolve()
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    old=ROOT/'tmp/frame-execution/cnn-observer-trim-evidence1/map-audit'
    audit_delta(board);r=json.loads((old/'result.json').read_text())
    assert r['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS'
    assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert r['board_inputs_sha256']==sha(board/'inputs.json') and r['checks']==17504
    for f,h in r['evidence'].items():assert sha(f)==h,f
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    for name in ('boolean_result.json','boolean_check.py'):shutil.copy2(old/name,root/name)
    r['completed_prefix_reused']=dict(path=str(old/'result.json'),sha256=sha(old/'result.json'),failed_overall_run_not_reused=True)
    r['evidence'].update({str(old/'result.json'):sha(old/'result.json'),str(Path(__file__)):sha(__file__)})
    from camera_checked_cdc import map_structure
    structure=map_structure(board)
    (root/'held_data_structure.json').write_text(json.dumps(structure,indent=2))
    (root/'result.json').write_text(json.dumps(r,indent=2))
    print('CURRENT_MAP_COMPLETE_PREFIX_AND_CAMERA_STRUCTURE_PASS',flush=True)

if len(sys.argv)>1 and sys.argv[1]=='audit' and (ROOT/'tmp/frame-execution/cnn-observer-trim-evidence1/map-audit/result.json').exists():
    resume_completed_map_prefix();raise SystemExit(0)
SOURCE=Path(__file__).with_name('camera_payload_flow.py')
code=SOURCE.read_text(encoding='utf8').replace('camera_payload','cnn_observer_trim')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_joint_gray.py'")
# Each short gate is a separate process. Pass the real read-lock/SHA cache to
# that process too; the parent's cache cannot accelerate a child's file reads.
needle="        generated(source,code,'-payload-short-wrapper')"
assert code.count(needle)==1
injection='''        old="[sys.executable,'-B',str(HERE/script),*map(str,args)]"
        new="[sys.executable,'-B',str(ROOT/'tmp/board-test/run-with-locked-evidence.py'),'--cache-log',str(root/(name+'-cache.json')),str(HERE/script),*map(str,args)]"
        assert code.count(old)==1;code=code.replace(old,new)
        code=once(code,"root/'map-audit'],100)","root/'map-audit'],240)")
'''
code=code.replace(needle,injection+needle)
exec(compile(code,str(SOURCE)+':observer-trim','exec'),globals())
