"""Replay an already frozen, bounded public-RTL test on the existing64-bit Questa.
No vendor/IP substitution, no assertion removal, no RTL changes.
"""
import argparse,json,os,re,shutil,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=('cluster','guard'),required=True);a=p.parse_args()
root=a.root.resolve();source=a.source.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
try:
 sim=Path('E:/quatra/questa_fse/win64')
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 meta=json.loads((source/'inputs.json').read_text());records=[]
 for row in meta['inputs']:
  old=Path(row['snapshot']);assert sha(old)==row['sha256'];dst=root/old.name;shutil.copy2(old,dst)
  records.append(dict(source=str(old),snapshot=str(dst),sha256=sha(dst)))
 if a.kind=='cluster':
  shutil.copytree(source/'vectors',root/'vectors')
  for f,h in meta['vectors'].items():assert sha(root/'vectors'/f)==h
  do=(source/'run.do').read_text().replace(source.as_posix(),root.as_posix())
  (root/'run.do').write_text(do)
 manifest=dict(inputs=records,source=str(source),source_inputs_sha256=sha(source/'inputs.json'),simulator_bin=str(sim),
  simulator_sha256=sha(sim/'vsim.exe'),kind=a.kind,runner_sha256=sha(__file__))
 if a.kind=='cluster':manifest['vectors']=meta['vectors'];manifest['do_sha256']=sha(root/'run.do')
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 run([sim/'vlib.exe','work'],root,'vlib.log')
 body,_=run([sim/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log',timeout=60)
 assert 'Errors: 0, Warnings: 0' in body
 top='tb_compute_column_cluster' if a.kind=='cluster' else 'tb_cnn_datapath_guard'
 args=[sim/'vsim.exe','-c']
 if a.kind=='cluster':
  args+=['-voptargs=+acc','work.'+top,'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+(root/'readback.hex').as_posix(),'-do','do run.do']
 else:args+=['-gACTIVE_TEST=1','work.'+top,'-do','onerror {quit -code 1}; run -all; quit -code 0']
 body,seconds=run(args,root,'simulation.log',timeout=240)
 assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',body)
 codes=re.findall(r'\*\* Warning: \(([^)]+)\)',body)
 if a.kind=='cluster':
  assert 'PASS RESIDUAL_ABORT_PROBE stages=5 direction_fault=1' in body
  assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body
  hold=re.findall(r'^# RESIDUAL_RETURN_HOLD checks=(\d+)$',body,re.M);assert len(hold)==1 and int(hold[0])>0
  lines=re.findall(r'^# (PASS COMPUTE_CLUSTER.*)$',body,re.M);assert len(lines)==1 and 'blocks=2 ' in lines[0]
  result=dict(status='DIRECT_SRAM_RESIDUAL_ABORT_FOUR_STATE_PASS',line=lines[0],abort_stages=5,residual_return_hold_checks=int(hold[0]))
 else:
  before=json.loads((ROOT/'tmp/frame-execution/shared-taps-guard-fourstate1/result.json').read_text())
  lines=re.findall(r'^# (PASS DATAPATH_GUARD.*)$',body,re.M);assert lines==[before['line']]
  active=re.findall(r'^# (PASS DATAPATH_ACTIVE_CANCEL.*)$',body,re.M);assert active==before['active_lines']
  result=dict(status='INTEGRATED_DATAPATH_GUARD_FOUR_STATE_PASS',line=lines[0],active_lines=active)
 # New simulator warning codes must be reviewed explicitly before final signoff.
 result.update(seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
  simulator_bin=str(sim),warning_codes=codes,warnings_reviewed=not codes,full_frame=False)
 for row in records:assert sha(row['snapshot'])==row['sha256']
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
