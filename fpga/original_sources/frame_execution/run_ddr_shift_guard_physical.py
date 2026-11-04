"""Full-system PLL setup fix: selected route and original-constraint evidence."""
import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_ddr_shift_guard_board import audit_delta,proof
from windows_process_tree import stop_launched_tree
from audit_canonical_multiseed import audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--reuse-startup',type=Path);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();delta=audit_delta(board,False);bound=proof(False)
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 stages=[]
 (root/'runner.py').write_bytes(Path(__file__).read_bytes())
 (root/'inputs.json').write_text(json.dumps(dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),delta=delta,proof=bound,seed=1,hardware_access=False,runner_sha256=sha(__file__)),indent=2))
 def run(name,script,args,budget):
  begin=time.monotonic()
  with (root/(name+'.log')).open('x') as log:
   proc=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row=dict(stage=name,pid=proc.pid,status='running',budget_s=budget);stages.append(row);(root/'stages.json').write_text(json.dumps(stages,indent=2))
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    (root/(name+'-cleanup.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
  row.update(status='finished',exit_code=code,seconds=time.monotonic()-begin);(root/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,code)
 try:
  if a.reuse_startup:
   previous=a.reuse_startup.resolve();s=json.loads((previous/'result.json').read_text());m=json.loads((previous/'inputs.json').read_text())
   assert s['status']=='FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS' and s['inputs_sha256']==sha(previous/'inputs.json')
   assert m['original_map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and s['simulation_sha256']==sha(previous/'simulation.log')
   for n,h in m['files'].items():assert sha(previous/n)==h
   shutil.copytree(previous,root/'startup')
   (root/'reused_startup.json').write_text(json.dumps(dict(root=str(previous),result_sha256=sha(previous/'result.json'),map_sha256=m['original_map_sha256']),indent=2))
  else:
   run('startup','validate_shift_guard_mapped.py',['--board',board,'--root',root/'startup'],180)
  physical=root/'pnr-seed1';canonical=root/'canonical-seed1';gray=root/'gray-seed1'
  reference=ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1'
  assert audit(reference)['current_timing_pass']
  run('physical','probe_shift_guard_physical.py',['--cnn-ram-reference',reference,'--base',board,'--root',physical,'--pnr-budget',1500,'--logic-limit',60800,'--optimization','CONGESTION_2','--fixed-dsp-layout','--seed',1,'--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json','--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],1950)
  pr=json.loads((physical/'result.json').read_text());assert all(r['exit_code']==0 for r in pr['stages'])
  import audit_control_state_implementation as impl
  impl.audit_delta=lambda b:audit_delta(b,False)
  implementation=impl.audit(board,physical)
  implementation['required_evidence'][str(HERE/'prepare_ddr_shift_guard_board.py')]=sha(HERE/'prepare_ddr_shift_guard_board.py')
  (root/'implementation.json').write_text(json.dumps(implementation,indent=2))
  run('canonical','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
  selected=audit(canonical);(root/'selected.json').write_text(json.dumps(selected,indent=2))
  assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
  run('gray','audit_snapshot_routed_gray.py',['--base',physical,'--root',gray],370)
  audit_delta(board,False)
  result=dict(status='SHIFT_GUARD_PHYSICAL_PASS_PENDING_VENDOR_REPLAY_AND_BUNDLE',selected=selected,gray=str(gray),hardware_access=False,board_download_allowed=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise

if __name__=='__main__':main()
