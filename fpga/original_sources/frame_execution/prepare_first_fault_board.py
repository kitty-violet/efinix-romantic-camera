"""Freeze an observation-only copy; bounded map, never access hardware."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from first_fault_observer import BASE,MODULE,transform_soc,transform_observer,audit_delta
from registered_ownership_board import audit_delta as audit_parent
from audit_control_state_map import audit as audit_state
from audit_snapshot_cells import audit as audit_sync
from windows_process_tree import stop_launched_tree

PROOF=ROOT/'tmp/frame-execution/first-fault-short1'

def check_proof():
    result=json.loads((PROOF/'result.json').read_text())
    assert result['status']=='FIRST_FAULT_FOUR_STATE_SHORT_PASS'
    assert sha(PROOF/'inputs.json')==result['inputs_sha256']
    assert sha(PROOF/'simulation.log')==result['log_sha256']
    data=json.loads((PROOF/'inputs.json').read_text())
    for name,h in data['files'].items():assert sha(PROOF/name)==h
    assert data['module_sha256']==sha(MODULE)
    assert data['transform_sha256']==sha(HERE/'first_fault_observer.py')
    assert data['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert (PROOF/'EfxSapphireSoc.v').read_text()==transform_soc((BASE/'assembly/EfxSapphireSoc.v').read_text())
    return dict(path=str(PROOF),result_sha256=sha(PROOF/'result.json'),inputs_sha256=sha(PROOF/'inputs.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--map-prepared',action='store_true');a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp');proof=check_proof();audit_parent(BASE)
    if not a.map_prepared:
        assert not root.exists();root.mkdir()
        for folder in ('rtl','assembly'):shutil.copytree(BASE/folder,root/folder)
        for f in BASE.iterdir():
            if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
        for name,fn in (('EfxSapphireSoc.v',transform_soc),('jtag_bringup_snapshot.sv',transform_observer)):
            (root/'assembly'/name).write_text(fn((BASE/'assembly'/name).read_text()),encoding='utf-8')
        meta=json.loads((BASE/'inputs.json').read_text())
        for rec in meta['inputs']:
            f=root/Path(rec['snapshot']).relative_to(BASE);rec.update(snapshot=str(f),sha256=sha(f))
        meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),first_fault_proof=proof,
                    first_fault_transform_sha256=sha(HERE/'first_fault_observer.py'),runner_sha256=sha(__file__),
                    hardware_access=False,board_download_allowed=False)
        (root/'inputs.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,root/'runner.py')
        (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
        print('FIRST_FAULT_BOARD_PREPARED_NOT_RELEASE');return
    audit_delta(root);assert not (root/'map.log').exists()
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (root/'map.log').open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (root/'progress.json').write_text(json.dumps(dict(stage='map',pid=proc.pid,budget_s=960)))
            try:code=proc.wait(timeout=960)
            except subprocess.TimeoutExpired:
                (root/'timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
        assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
        net=root/'outflow/ti60f225_oob.map.v'
        (root/'control_state_map_audit.json').write_text(json.dumps(audit_state(net,'cpu/','process_clk_150'),indent=2))
        sync=audit_sync(net.read_text(),prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/',system='process_clk_150',tck='jtag_inst1_TCK')
        (root/'snapshot_sync_audit.json').write_text(json.dumps(sync,indent=2))
        warnings=(root/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        old=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
        assert not set(re.findall(pattern,warnings,re.M))-set(re.findall(pattern,old,re.M))
        assert 'EFX-0654' not in warnings
        rpt=root/'outflow/ti60f225_oob.map.rpt'
        resources={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
        audit_delta(root)
        result=dict(status='FIRST_FAULT_FULL_BOARD_MAP_NOT_RELEASE',seconds=time.monotonic()-start,resources=resources,
                    inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),
                    physical_pass=False,hardware_access=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start),indent=2));raise
if __name__=='__main__':main()
