"""Reuse verified successful gates, then serialize the single-seat simulator."""
import json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
from camera_joint_board import BOARD
OLD=ROOT/'tmp/frame-execution/camera-joint-evidence1'
DEST=ROOT/'tmp/frame-execution/camera-joint-evidence2'
def save(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 assert not DEST.exists();DEST.mkdir();stages=[];mapped=sha(BOARD/'outflow/ti60f225_oob.map.v')
 oldstages=json.loads((OLD/'stages.json').read_text());assert [s['exit_code'] for s in oldstages[:3]]==[0,0,0]
 assert oldstages[-1]['stage']=='observe' and oldstages[-1]['exit_code']==1
 failure=(OLD/'observe.log').read_text();assert 'only one session is allowed' in failure and 'Error loading design' in failure
 assert json.loads((OLD.parent/(OLD.name+'-adapter')/'inputs.json').read_text())['original_map_sha256']==mapped
 assert json.loads((OLD/'startup/inputs.json').read_text())['original_map_sha256']==mapped
 # Copy only already completed evidence, with byte hashes recorded; no PASS
 # is inherited from the failed observation stage and nothing is rerun twice.
 reused={}
 for source,target in [(OLD/n,DEST/n) for n in ('map-audit','rtl','startup')]+[(OLD.parent/(OLD.name+'-adapter'),DEST.parent/(DEST.name+'-adapter'))]:
  assert not target.exists();shutil.copytree(source,target)
  for f in source.rglob('*'):
   if f.is_file():
    d=target/f.relative_to(source);assert sha(d)==sha(f);reused[str(f)]=sha(f)
  stages.append(dict(stage='reused_'+source.name,status='finished',exit_code=0,source=str(source),target=str(target)))
 save(DEST/'reused_evidence.json',dict(status='SUCCESSFUL_SAME_MAP_PREFIX_REUSED',map_sha256=mapped,evidence=reused,old_stages_sha256=sha(OLD/'stages.json'),old_failure_sha256=sha(OLD/'observe.log'),runner_sha256=sha(__file__)))
 def run(name,args,budget):
  with (DEST/(name+'.log')).open('x') as log:
   start=time.monotonic();child=subprocess.Popen([sys.executable,'-B',str(HERE/'camera_joint_flow.py'),'flow',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row=dict(stage=name,pid=child.pid,status='running',budget_s=budget);stages.append(row);save(DEST/'stages.json',stages)
   try:rc=child.wait(timeout=budget)
   except subprocess.TimeoutExpired:save(DEST/(name+'-timeout.json'),stop_launched_tree(child));raise
   row.update(status='finished',exit_code=rc,seconds=time.monotonic()-start);save(DEST/'stages.json',stages);assert rc==0,(name,rc)
 startup=DEST/'startup';adapter=DEST.parent/(DEST.name+'-adapter')
 run('observe',['observe','--source',startup,'--root',DEST/'observe'],180)
 serial_parent=DEST.parent/(DEST.name+'-serial');serial=serial_parent/'new'
 run('serial',['serial','--board',BOARD,'--startup',startup,'--root',serial_parent,'--tck-fast'],300)
 evidence={str(DEST/'reused_evidence.json'):sha(DEST/'reused_evidence.json')}
 for folder,status in ((startup,'COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'),(DEST/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),(serial,'MAPPED_SERIAL_APB_OFS_RESET_PASS'),(adapter,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
  r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
  assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
  assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped
  assert r.get('log_sha256',r.get('simulation_sha256'))==sha(folder/'simulation.log')
  for n,h in i['files'].items():assert sha(folder/n)==h
  for n in ('result.json','inputs.json','simulation.log'):evidence[str(folder/n)]=sha(folder/n)
 for n in ('progress_counter_checks.json','paired_reply_result.json'):
  shutil.copy2(serial_parent/n,serial/n);evidence[str(serial/n)]=sha(serial/n)
 assert json.loads((serial/'inputs.json').read_text())['tck_hz']==10000000
 save(DEST/'result.json',dict(status='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS',map_sha256=mapped,board_inputs_sha256=sha(BOARD/'inputs.json'),stages=stages,evidence=evidence,runner_sha256=sha(__file__),single_license_serial=True,hardware_access=False,board_release=False))
 print('JOINT_ALL_SHORTS_PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(DEST/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
