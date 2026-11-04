"""Serial same-map short gates; no full-frame simulation or board access."""
import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree

def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 root.mkdir();stages=[];mapped=sha(board/'outflow/ti60f225_oob.map.v')
 def run(name,script,args,budget):
  start=time.monotonic()
  with (root/(name+'.log')).open('x') as f:
   child=subprocess.Popen([sys.executable,'-B',str(HERE/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   row=dict(stage=name,pid=child.pid,status='running',budget_s=budget);stages.append(row);save(root/'stages.json',stages)
   try:rc=child.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    save(root/(name+'-timeout.json'),stop_launched_tree(child));raise
  row.update(status='finished',exit_code=rc,seconds=time.monotonic()-start);save(root/'stages.json',stages)
  assert rc==0,(name,rc)
 try:
  run('map-audit','audit_split_logic_outputs_map.py',['--board',board,'--root',root/'map-audit'],100)
  run('rtl','split_logic_outputs_flow.py',['rtl','--board',board,'--root',root/'rtl'],90)
  adapter=root.parent/(root.name+'-adapter')
  run('adapter','validate_control_adapter_idle_mapped.py',['--board',board,'--tag',adapter.name],250)
  # This gate already covers startup and three resets, as well as the real
  # adapter. Reuse its identical map/model/stimulus/log rather than rerun it.
  sr=json.loads((adapter/'result.json').read_text());si=json.loads((adapter/'inputs.json').read_text())
  assert sr['status']=='ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS'
  assert sr['inputs_sha256']==sha(adapter/'inputs.json') and si['original_map_sha256']==mapped
  assert sr['simulation_sha256']==sha(adapter/'simulation.log')
  body=(adapter/'tb.sv').read_text()
  for check in ('adapter idle','startup boot','unexpected wait-loop state','polling made no progress','MAPPED_STARTUP_RECOVERY_PASS'):assert check in body
  startup=root/'startup';startup.mkdir()
  for n,h in si['files'].items():
   assert sha(adapter/n)==h;shutil.copy2(adapter/n,startup/n)
  shutil.copy2(adapter/'simulation.log',startup/'simulation.log')
  meta=dict(si,source_replay=str(adapter),source_inputs_sha256=sha(adapter/'inputs.json'),
   source_result_sha256=sha(adapter/'result.json'),collator_sha256=sha(__file__),
   scope=si['scope']+'; same mapped startup/three-reset superset, with real adapter FSMs; protected reset modeled')
  save(startup/'inputs.json',meta)
  save(startup/'result.json',dict(sr,status='COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT',
   inputs_sha256=sha(startup/'inputs.json'),startup_checks_reused_from_passing_superset=True))
  run('observe','split_logic_outputs_flow.py',['observe','--source',startup,'--root',root/'observe'],180)
  serial_parent=root.parent/(root.name+'-serial');serial=serial_parent/'new'
  run('serial','split_logic_outputs_flow.py',['serial','--board',board,'--startup',startup,'--root',serial_parent,'--tck-fast'],300)
  evidence={}
  for folder,status in ((startup,'COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'),
   (root/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),(serial,'MAPPED_SERIAL_APB_OFS_RESET_PASS'),
   (adapter,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
   r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
   assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
   assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped
   assert r.get('log_sha256',r.get('simulation_sha256'))==sha(folder/'simulation.log')
   for n,h in i['files'].items():assert sha(folder/n)==h
   for n in ('result.json','inputs.json','simulation.log'):evidence[str(folder/n)]=sha(folder/n)
  for n in ('progress_counter_checks.json','paired_reply_result.json'):
   assert not (serial/n).exists();shutil.copy2(serial_parent/n,serial/n);evidence[str(serial/n)]=sha(serial/n)
  assert json.loads((serial/'inputs.json').read_text())['tck_hz']==10000000
  save(root/'result.json',dict(status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS',map_sha256=mapped,
   board_inputs_sha256=sha(board/'inputs.json'),stages=stages,evidence=evidence,runner_sha256=sha(__file__),
   hardware_access=False,board_release=False,ddr_phy_calibration_proven=False))
 except Exception as exc:
  save(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False));raise

if __name__=='__main__':main()
