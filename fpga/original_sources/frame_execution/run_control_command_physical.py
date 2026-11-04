"""Full command-repair gates, short tests then selected-seed implementation."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import audit_delta
from audit_command_toggle_sync import audit as command_sync
from audit_canonical_multiseed import audit as canonical_audit
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();delta=audit_delta(board)
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 mapped=json.loads((board/'result.json').read_text());net=board/'outflow/ti60f225_oob.map.v'
 assert mapped['inputs_sha256']==sha(board/'inputs.json') and mapped['map_sha256']==sha(net)
 assert mapped['new_undriven_signatures']==[] and not mapped['comb_loop_warning']
 sync=command_sync(net.read_text());assert not sync['errors'],sync
 root.mkdir();(root/'command_sync.json').write_text(json.dumps(sync,indent=2))
 (root/'runner.py').write_bytes(Path(__file__).read_bytes())
 (root/'inputs.json').write_text(json.dumps(dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),map_sha256=sha(net),delta=delta,runner_sha256=sha(__file__),hardware_access=False,board_release=False),indent=2))
 stages=[]
 def run(name,script,args,budget):
  started=time.monotonic()
  with (root/(name+'.log')).open('x') as log:
   proc=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row=dict(name=name,pid=proc.pid,status='running',budget_s=budget);stages.append(row);(root/'stages.json').write_text(json.dumps(stages,indent=2))
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    (root/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
  row.update(status='finished',exit_code=code,seconds=time.monotonic()-started);(root/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,code)
 try:
  run('startup','validate_command_progress_mapped.py',['--board',board,'--root',root/'startup'],180)
  serial=root.parent/(root.name+'-serial')
  run('serial','validate_command_progress_serial.py',['--startup',root/'startup','--root',serial],240)
  reference=ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1'
  assert canonical_audit(reference)['current_timing_pass']
  physical=root/'pnr-seed1';canonical=root/'canonical-seed1';gray=root/'gray-seed1'
  run('physical','probe_shift_guard_physical.py',['--cnn-ram-reference',reference,'--base',board,'--root',physical,
   '--pnr-budget',1500,'--logic-limit',60800,'--optimization','CONGESTION_2','--fixed-dsp-layout','--seed',1,
   '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'],1950)
  import audit_control_state_implementation as impl
  impl.audit_delta=audit_delta
  (root/'implementation.json').write_text(json.dumps(impl.audit(board,physical),indent=2))
  run('canonical','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
  selected=canonical_audit(canonical);(root/'selected.json').write_text(json.dumps(selected,indent=2))
  assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
  run('gray','audit_command_progress_gray.py',['--base',physical,'--root',gray],380)
  audit_delta(board)
  (root/'result.json').write_text(json.dumps(dict(status='COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE',selected=selected,serial=str(serial),gray=str(gray),hardware_access=False,board_release=False),indent=2))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
