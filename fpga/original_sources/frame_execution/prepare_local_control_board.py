"""Whole board with cycle-transparent CNN-local controls, original global map settings."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_control_state_board import audit_delta as audit_parent
from audit_control_state_map import audit as audit_state
from audit_snapshot_cells import audit as audit_sync
from localize_window_controls import localize
from windows_process_tree import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/connected-board-control-state1'
PROOFS=ROOT/'tmp/frame-execution/window-local-controls-sim3'
SMALL=ROOT/'tmp/frame-execution/window-local-controls-map2'

def audit_delta(board):
    meta=json.loads((board/'inputs.json').read_text());changed=[]
    for row in meta['inputs']:
        rel=Path(row['snapshot']).relative_to(board);assert sha(board/rel)==row['sha256']
        if sha(board/rel)!=sha(BASE/rel):changed.append(rel.as_posix())
    assert changed==['assembly/fixed_window_dot8x32.sv'],changed
    assert (board/changed[0]).read_text()==localize((BASE/changed[0]).read_text())
    return dict(changed=changed,added_clock_cycles=0,pins_clocks_reset_sdc_ddr_unchanged=True,
                global_synthesis_settings_unchanged=True,hardware_access=False)

def audit_branches(net):
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',net.read_text(),flags=re.S)
    prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/arithmetic/'
    drivers=[];dsp=[]
    for kind,name,body in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\)\s*;',text,re.M|re.S):
        name=name.lstrip('\\')
        if not name.startswith(prefix):continue
        ports=dict(re.findall(r'\.(\w+)\(([^()]*)\)',body))
        if '_driver' in name:drivers.append(dict(kind=kind,name=name,ports=ports))
        if kind=='EFX_DSP24':dsp.append(ports)
    assert 62<=len(drivers)<=63,len(drivers)
    assert all(d['kind']=='EFX_LUT4' for d in drivers)
    assert len({d['ports']['O'] for d in drivers})==len(drivers)
    assert len(dsp)==256 and all('pair_advance' in d['CE'] for d in dsp)
    return dict(status='LOCAL_CNN_BRANCHES_RETAINED',drivers=drivers,dsp24=256,map_sha256=sha(net))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();audit_parent(BASE)
    proof=json.loads((PROOFS/'result.json').read_text());assert proof['status']=='FIXED_WINDOW_DOT8X32_OFFICIAL_ARITHMETIC_PASS'
    assert sha(PROOFS/'inputs.json')==proof['inputs_sha256'] and len(proof['runs'])==3
    expected=localize((BASE/'assembly/fixed_window_dot8x32.sv').read_text())
    assert (PROOFS/'fixed_window_dot8x32.sv').read_text()==expected
    small=json.loads((SMALL/'result.json').read_text());assert small['status']=='LOCAL_WINDOW_63_BRANCHES_256_DSP_CE_STRUCTURAL_PASS'
    assert small['source_sha256']==sha(PROOFS/'fixed_window_dot8x32.sv')
    for r in json.loads((PROOFS/'inputs.json').read_text())['inputs']:assert sha(r['snapshot'])==r['sha256']
    for r in proof['runs']:assert sha(PROOFS/f'simulation_{r["seed"]}.log')==r['log_sha256']
    root.mkdir()
    for folder in ('rtl','assembly'):shutil.copytree(BASE/folder,root/folder)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
    (root/'assembly/fixed_window_dot8x32.sv').write_text(expected)
    meta=json.loads((BASE/'inputs.json').read_text())
    for r in meta['inputs']:
        rel=Path(r['snapshot']).relative_to(BASE);r['snapshot']=str(root/rel);r['sha256']=sha(root/rel)
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),local_short_proofs={str(PROOFS):sha(PROOFS/'result.json'),str(SMALL):sha(SMALL/'result.json')},
                local_transform_sha256=sha(HERE/'localize_window_controls.py'),runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,root/'runner.py')
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 1200\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (root/'map.log').open('x') as log:
            child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (root/'progress.json').write_text(json.dumps(dict(stage='map',pid=child.pid,host_budget_s=1260)))
            try:code=child.wait(timeout=1260)
            except subprocess.TimeoutExpired:stop_launched_tree(child);raise
        assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
        net=root/'outflow/ti60f225_oob.map.v'
        audits={
          'local_branches_audit.json':audit_branches(net),
          'control_state_map_audit.json':audit_state(net,'cpu/','process_clk_150'),
          'snapshot_sync_audit.json':audit_sync(net.read_text(),prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/',system='process_clk_150',tck='jtag_inst1_TCK')}
        for name,result in audits.items():(root/name).write_text(json.dumps(result,indent=2))
        ws=(root/'outflow/ti60f225_oob.warn.log').read_text(errors='replace');old=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
        assert not set(re.findall(pattern,ws,re.M))-set(re.findall(pattern,old,re.M))
        assert 'EFX-0654' not in ws
        rpt=root/'outflow/ti60f225_oob.map.rpt';audit_delta(root)
        result=dict(status='LOCAL_CONTROL_FULL_BOARD_MAP_NOT_RELEASE',seconds=time.monotonic()-start,
            inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),
            audit_sha256={n:sha(root/n) for n in audits},snapshot_sync_audit_sha256=sha(root/'snapshot_sync_audit.json'),
            physical_pass=False,hardware_access=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start),indent=2));raise
if __name__=='__main__':main()
