"""Exercise error classification and owned Windows child-tree cleanup without EDA."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,sha
from run_board_frame import failure_kind,stop_spawned_tree
from owned_simulator_process import spawn_owned,close_owned

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 cases=[('# tb_board_startup.io_model.memory.chk_err: at time 99 ps ERROR: tWTR violation','DDR_DEVICE_TIMING_ERROR'),
        ('# ** Warning: Failed to open readmem file ddr3_controller.bin in read mode.','MISSING_INITIALIZATION_IMAGE'),
        ('# ** Fatal: broken test','SIMULATOR_ERROR'),('# ** Error: elaboration failed','SIMULATOR_ERROR'),
        ('# ** Warning: optional input unused',None),('# BOARD_PROGRESS errors=0000',None),('Errors: 0, Warnings: 0',None)]
 for line,want in cases:assert failure_kind(line)==want
 child_code="import time; print('CHILD_RUNNING',flush=True); time.sleep(60)"
 parent_code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-u','-c',sys.argv[1]]); print('OWNED_CHILD_PID='+str(p.pid),flush=True); time.sleep(60)"
 start=time.monotonic()
 proc=spawn_owned([sys.executable,'-u','-c',parent_code,child_code],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 lines=[proc.stdout.readline().strip() for _ in range(2)]
 first=next(line for line in lines if line.startswith('OWNED_CHILD_PID='))
 child=int(first.split('=')[1]);assert 'CHILD_RUNNING' in lines
 # Keep an unrelated sentinel alive: tree cleanup must never kill by image name.
 sentinel=spawn_owned([sys.executable,'-c','import time; time.sleep(60)'])
 try:
  stopped=stop_spawned_tree(proc);proc.communicate(timeout=10)
  assert proc.returncode!=0 and sentinel.poll() is None
  # communicate returned EOF although the child inherited stdout: no orphaned
  # child can keep the pipe open. The separately owned sentinel remains alive.
  assert stop_spawned_tree(proc) is None
 finally:
  if proc.poll() is None:stop_spawned_tree(proc)
  stop_spawned_tree(sentinel);sentinel.wait(timeout=10)
  close_owned(proc);close_owned(sentinel)
 result=dict(status='BOARD_FAILFAST_CLASSIFICATION_AND_TREE_CLEANUP_PASS',classification_cases=len(cases),
  owned_parent=proc.pid,owned_child=child,unrelated_sentinel_survived=True,termination=stopped,
  seconds=time.monotonic()-start,runner_sha256=sha(Path(__file__).with_name('run_board_frame.py')))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
