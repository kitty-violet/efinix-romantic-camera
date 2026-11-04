"""Isolated read-only observation delta on the actual, released diagnostic board."""
import argparse,collections,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_local_reset_board import audit_delta as audit_parent
from control_snapshot_integration import instrument_soc,instrument_top
from windows_process_tree import stop_launched_tree
from audit_snapshot_cells import audit as audit_sync_cells
BASE=ROOT/'tmp/frame-execution/connected-board-local-reset1'
PROOF=ROOT/'tmp/frame-execution/integrated-control-snapshot-short4'
SNAPSHOT=HERE/'variants/board_control_snapshot/jtag_bringup_snapshot.sv'
ADD='        <efx:design_file name="assembly/jtag_bringup_snapshot.sv" version="default" library="default" />\n'

def audit_delta(board):
 meta=json.loads((board/'inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 assert (board/'assembly/EfxSapphireSoc.v').read_text()==instrument_soc((BASE/'assembly/EfxSapphireSoc.v').read_text())
 assert (board/'rtl/ti60f225_oob_top.v').read_text()==instrument_top((BASE/'rtl/ti60f225_oob_top.v').read_text())
 assert sha(board/'assembly/jtag_bringup_snapshot.sv')==sha(SNAPSHOT)==sha(PROOF/SNAPSHOT.name)
 xml=(board/'ti60f225_oob.xml').read_text();assert xml.count(ADD)==1
 assert xml.replace(ADD,'')==(BASE/'ti60f225_oob.xml').read_text()
 changed=[]
 for row in meta['inputs']:
  f=Path(row['snapshot']);rel=f.relative_to(board)
  if not (BASE/rel).exists() or sha(f)!=sha(BASE/rel):changed.append(rel.as_posix())
 assert set(changed)=={'assembly/EfxSapphireSoc.v','rtl/ti60f225_oob_top.v','assembly/jtag_bringup_snapshot.sv','ti60f225_oob.xml'}
 for rel in ('ti60f225_oob.sdc','ti60f225_oob.peri.xml','ddr516_reference.sv','ddr516_v4_controller.sv'):
  assert sha(board/rel)==sha(BASE/rel),rel
 assert sha(board/'assembly/EfxSapphireSoc.v')==sha(PROOF/'EfxSapphireSoc.v')
 assert sha(board/'rtl/ti60f225_oob_top.v')==sha(PROOF/'candidate_board_top.v')
 return dict(changed=changed,pins_clocks_sdc_ddr_unchanged=True,short_bound=True,
             functional_commands_unchanged=True,hardware_access=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 parent=audit_parent(BASE);proof=json.loads((PROOF/'result.json').read_text())
 assert proof['status']=='INTEGRATED_REAL_CONTROL_SNAPSHOT_SHORT_PASS' and proof['evidence_sha256']==sha(PROOF/'inputs.json')
 evidence=json.loads((PROOF/'inputs.json').read_text())
 for f,h in evidence['original'].items():assert sha(f)==h
 for f,h in evidence['files'].items():assert sha(PROOF/f)==h
 root.mkdir()
 for name in ('rtl','assembly'):shutil.copytree(BASE/name,root/name)
 for f in BASE.iterdir():
  if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
 (root/'assembly/EfxSapphireSoc.v').write_text(instrument_soc((BASE/'assembly/EfxSapphireSoc.v').read_text()),encoding='utf-8')
 (root/'rtl/ti60f225_oob_top.v').write_text(instrument_top((BASE/'rtl/ti60f225_oob_top.v').read_text()),encoding='utf-8')
 shutil.copy2(SNAPSHOT,root/'assembly'/SNAPSHOT.name)
 xml=root/'ti60f225_oob.xml';body=xml.read_text();needle='        <efx:top_module name="ti60f225_oob_top" />\n';assert body.count(needle)==1
 xml.write_text(body.replace(needle,needle+ADD),encoding='utf-8')
 meta=json.loads((BASE/'inputs.json').read_text())
 for row in meta['inputs']:
  rel=Path(row['snapshot']).relative_to(BASE);row['snapshot']=str(root/rel);row['sha256']=sha(root/rel)
 meta['inputs'].append(dict(source=str(SNAPSHOT),snapshot=str(root/'assembly'/SNAPSHOT.name),sha256=sha(SNAPSHOT)))
 meta.update(parent=str(BASE),parent_audit=parent,observation_proof=str(PROOF),observation_proof_sha256=sha(PROOF/'result.json'),
             observation_runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
 (root/'inputs.json').write_text(json.dumps(meta,indent=2));(root/'runner.py').write_bytes(Path(__file__).read_bytes())
 (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
 if a.prepare_only:print('PREPARED',root);return
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 1200\nexit /b %errorlevel%\n')
 started=time.monotonic()
 try:
  with (root/'map.log').open('w') as log:
   proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   (root/'progress.json').write_text(json.dumps(dict(stage='map',pid=proc.pid,host_budget_s=1260)))
   try:code=proc.wait(timeout=1260)
   except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
  assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
  warnings=root/'outflow/ti60f225_oob.warn.log';ws=warnings.read_text(errors='replace')
  old=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
  pat=r'^\[EFX-0201 WARNING\] (.*?)\. \('
  assert not set(re.findall(pat,ws,re.M))-set(re.findall(pat,old,re.M))
  assert 'EFX-0654' not in ws
  net=root/'outflow/ti60f225_oob.map.v';assert 'observation' in net.read_text()
  sync=audit_sync_cells(net.read_text(),prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/',system='process_clk_150',tck='jtag_inst1_TCK')
  (root/'snapshot_sync_audit.json').write_text(json.dumps(sync,indent=2))
  report=root/'outflow/ti60f225_oob.map.rpt'
  resources={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',report.read_text(),re.M)}
  audit_delta(root)
  result=dict(status='OBSERVATION_FULL_BOARD_MAP_NOT_RELEASE',resources=resources,seconds=time.monotonic()-started,
    inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(report),warning_sha256=sha(warnings),
    snapshot_sync_audit_sha256=sha(root/'snapshot_sync_audit.json'),physical_timing_pass=False,hardware_access=False,board_download_allowed=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started),indent=2));raise
if __name__=='__main__':main()
