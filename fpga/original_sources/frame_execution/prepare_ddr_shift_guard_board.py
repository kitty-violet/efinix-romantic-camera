"""Freeze/map the full deployed system with only the public PLL setup fix."""
import argparse,json,re,shutil,subprocess,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree
from first_fault_observer import audit_delta as audit_parent
from audit_control_state_map import audit as audit_state
from audit_snapshot_cells import audit as audit_sync

BASE=ROOT/'tmp/frame-execution/connected-board-first-fault1'
CAL=ROOT/'tmp/frame-execution/ddr-shift-guard-calibration3'
UNIT=ROOT/'tmp/frame-execution/ddr-shift-guard-unit1'

def proof(require_calibration=True):
 m=json.loads((CAL/'inputs.json').read_text())
 if require_calibration:
  r=json.loads((CAL/'result.json').read_text())
  assert r['status']=='GUARDED_SHIFT_VENDOR_CALIBRATION_RW_PASS' and r['violations']==0
  assert sha(CAL/'inputs.json')==r['inputs_sha256'] and sha(CAL/'simulation.log')==r['log_sha256']
  if 'post_audit' in r:
   assert sha(CAL/'failure.json')==r['post_audit']['original_failure_sha256']
   assert sha(HERE/'audit_ddr_shift_completed_log.py')==r['post_audit']['auditor_sha256']
 for name,h in m['files'].items():assert sha(CAL/name)==h,name
 u=json.loads((UNIT/'result.json').read_text());assert u['status']=='PLL_SHIFT_GUARD_FOURSTATE_PASS'
 assert sha(UNIT/'simulation.log')==u['log_sha256']
 for name,h in u['inputs'].items():assert sha(UNIT/name)==sha(HERE/name)==h
 assert sha(CAL/'ddr516_pll_shift_guard.sv')==sha(HERE/'ddr516_pll_shift_guard.sv')
 return dict(calibration=str(CAL),calibration_inputs_sha256=sha(CAL/'inputs.json'),calibration_completion_required_for_release=True,unit=str(UNIT),unit_result_sha256=sha(UNIT/'result.json'))

def audit_delta(board,require_calibration=True):
 parent=audit_parent(BASE);p=proof(require_calibration);meta=json.loads((board/'inputs.json').read_text())
 assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
 assert meta['shift_guard_proof']==p
 bm=json.loads((BASE/'inputs.json').read_text());changed=[]
 for row in bm['inputs']:
  rel=Path(row['snapshot']).relative_to(BASE);old=BASE/rel;new=board/rel
  assert sha(old)==row['sha256'],old
  if sha(new)!=sha(old):changed.append(str(rel))
 assert set(changed)=={'ddr516_v4_controller.sv','ti60f225_oob.xml'},changed
 assert sha(board/'ddr516_v4_controller.sv')==sha(CAL/'ddr516_v4_controller.sv')
 assert sha(board/'ddr516_pll_shift_guard.sv')==sha(HERE/'ddr516_pll_shift_guard.sv')
 marker='        <efx:design_file name="ddr516_pll_shift_guard.sv" version="default" library="default" />\n'
 assert (board/'ti60f225_oob.xml').read_text().replace(marker,'')==(BASE/'ti60f225_oob.xml').read_text()
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 return dict(status='FULL_SYSTEM_ONLY_PLL_SHIFT_SETUP_DELTA',parent=str(BASE),parent_audit=parent,
             changed=changed,added=['ddr516_pll_shift_guard.sv'],proof=p,hardware_access=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--map-prepared',action='store_true');a=p.parse_args();root=a.root.resolve()
 # Synthesis is independent and may overlap the vendor replay. Release is not.
 assert root.is_relative_to(ROOT/'tmp');p=proof(False);audit_parent(BASE)
 if not a.map_prepared:
  assert not root.exists();root.mkdir()
  for folder in ('rtl','assembly'):shutil.copytree(BASE/folder,root/folder)
  for f in BASE.iterdir():
   if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
  for name in ('ddr516_pll_shift_guard.sv','ddr516_v4_controller.sv'):shutil.copy2(CAL/name,root/name)
  xml=root/'ti60f225_oob.xml';body=xml.read_text();marker='        <efx:design_file name="ddr516_pll_shift_guard.sv" version="default" library="default" />\n'
  assert body.count('        <efx:top_module')==1
  # Efinity's schema requires top_module before the design_file sequence.
  body=body.replace('        <efx:design_file',marker+'        <efx:design_file',1)
  design=ET.fromstring(body).find('{http://www.efinixinc.com/enf_proj}design_info')
  assert list(design)[0].tag.endswith('}top_module')
  xml.write_text(body)
  meta=json.loads((BASE/'inputs.json').read_text())
  for row in meta['inputs']:
   dst=root/Path(row['snapshot']).relative_to(BASE);row.update(snapshot=str(dst),sha256=sha(dst))
  meta['inputs'].append(dict(source=str(HERE/'ddr516_pll_shift_guard.sv'),snapshot=str(root/'ddr516_pll_shift_guard.sv'),sha256=sha(root/'ddr516_pll_shift_guard.sv')))
  meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),shift_guard_proof=p,hardware_access=False,board_download_allowed=False,runner_sha256=sha(__file__))
  (root/'inputs.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,root/'runner.py')
  (root/'delta_audit.json').write_text(json.dumps(audit_delta(root,False),indent=2));print('SHIFT_GUARD_FULL_BOARD_PREPARED_NOT_RELEASED');return
 audit_delta(root,False);assert not (root/'map.log').exists()
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
  warnings=(root/'outflow/ti60f225_oob.warn.log').read_text(errors='replace');old=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
  pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
  assert not set(re.findall(pattern,warnings,re.M))-set(re.findall(pattern,old,re.M));assert 'EFX-0654' not in warnings
  report=root/'outflow/ti60f225_oob.map.rpt';resources={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',report.read_text(),re.M)}
  audit_delta(root,False)
  result=dict(status='SHIFT_GUARD_FULL_BOARD_MAP_NOT_RELEASE',seconds=time.monotonic()-start,resources=resources,inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(report),hardware_access=False,board_download_allowed=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
