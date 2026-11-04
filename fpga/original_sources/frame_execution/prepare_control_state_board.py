"""One-source control-encoding candidate on the frozen observation board."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_snapshot_board import audit_delta as audit_parent
from control_state_specialization import specialize
from audit_control_state_map import audit as audit_state
from audit_snapshot_cells import audit as audit_sync
from windows_process_tree import stop_launched_tree
BASE=ROOT/'tmp/frame-execution/connected-board-control-snapshot2'
PROOFS=[ROOT/'tmp/frame-execution'/p for p in ('control-state-lockstep-smoke3','control-state-lockstep-a1','control-state-mapped-smoke4')]

def audit_delta(board):
    meta=json.loads((board/'inputs.json').read_text())
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    expected=specialize((BASE/'assembly/EfxSapphireSoc.v').read_text())
    assert (board/'assembly/EfxSapphireSoc.v').read_text()==expected
    changed=[]
    for row in meta['inputs']:
        rel=Path(row['snapshot']).relative_to(board)
        if sha(board/rel)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert changed==['assembly/EfxSapphireSoc.v'],changed
    for rel in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','rtl/ti60f225_oob_top.v','ddr516_reference.sv','ddr516_v4_controller.sv'):
        assert sha(board/rel)==sha(BASE/rel),rel
    return dict(changed=changed,control_state_encoding_changed=True,commands_and_cycle_behavior_short_proven=True,
                pins_clocks_reset_sdc_ddr_unchanged=True,physical_root_cause_proven=False,hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE)
    for proof in PROOFS:
        result=json.loads((proof/'result.json').read_text());assert result['status'] in ('CONTROL_STATE_LOCKSTEP_PASS','ACTUAL_MAPPED_CONTROL_LOCKSTEP_PASS')
        assert sha(proof/'inputs.json')==result['inputs_sha256']
        for name,h in json.loads((proof/'inputs.json').read_text())['files'].items():assert sha(proof/name)==h
    assert json.loads((PROOFS[1]/'result.json').read_text())['full_a_load']
    candidate=specialize((BASE/'assembly/EfxSapphireSoc.v').read_text())
    assert candidate==(PROOFS[0]/'EfxSapphireSoc.v').read_text()==(PROOFS[1]/'EfxSapphireSoc.v').read_text()
    root.mkdir()
    for folder in ('rtl','assembly'):shutil.copytree(BASE/folder,root/folder)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
    (root/'assembly/EfxSapphireSoc.v').write_text(candidate,encoding='utf-8')
    meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        rel=Path(row['snapshot']).relative_to(BASE);row['snapshot']=str(root/rel);row['sha256']=sha(root/rel)
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),control_state_proofs={str(p):sha(p/'result.json') for p in PROOFS},
                transform_sha256=sha(HERE/'control_state_specialization.py'),control_state_runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,root/'runner.py')
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 1200\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (root/'map.log').open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (root/'progress.json').write_text(json.dumps(dict(stage='map',pid=proc.pid,host_budget_s=1260)))
            try:code=proc.wait(timeout=1260)
            except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
        assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
        net=root/'outflow/ti60f225_oob.map.v';state=audit_state(net,'cpu/','process_clk_150')
        (root/'control_state_map_audit.json').write_text(json.dumps(state,indent=2))
        sync=audit_sync(net.read_text(),prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/',system='process_clk_150',tck='jtag_inst1_TCK')
        (root/'snapshot_sync_audit.json').write_text(json.dumps(sync,indent=2))
        ws=(root/'outflow/ti60f225_oob.warn.log').read_text(errors='replace');old=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
        assert not set(re.findall(pattern,ws,re.M))-set(re.findall(pattern,old,re.M))
        assert 'EFX-0654' not in ws
        rpt=root/'outflow/ti60f225_oob.map.rpt'
        resources={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
        audit_delta(root)
        result=dict(status='CONTROL_STATE_FULL_BOARD_MAP_NOT_RELEASE',seconds=time.monotonic()-start,resources=resources,
                    inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),
                    control_state_audit_sha256=sha(root/'control_state_map_audit.json'),snapshot_sync_audit_sha256=sha(root/'snapshot_sync_audit.json'),
                    physical_pass=False,hardware_access=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start),indent=2));raise
if __name__=='__main__':main()
