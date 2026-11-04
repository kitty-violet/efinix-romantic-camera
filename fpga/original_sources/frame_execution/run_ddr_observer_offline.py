"""One bounded sequential offline pipeline; never downloads or opens USB."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
from ddr_cal_observer import BASE,audit_delta
from prepare_ddr_observer_board import check_proof
from audit_canonical_multiseed import audit
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--seed',type=int,default=1);p.add_argument('--reuse-short-batch',type=Path);a=p.parse_args()
 b=a.board.resolve();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 audit_delta(b);proof=check_proof();stages=[]
 (r/'runner.py').write_bytes(Path(__file__).read_bytes())
 (r/'inputs.json').write_text(json.dumps(dict(board=str(b),board_inputs_sha256=sha(b/'inputs.json'),proof=proof,
   seed=a.seed,runner_sha256=sha(__file__),hardware_access=False),indent=2))
 def run(name,script,args,budget):
  start=time.monotonic();log=r/(name+'.log')
  with log.open('x') as output:
   child=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
   row=dict(stage=name,pid=child.pid,status='running',budget_s=budget);stages.append(row)
   (r/'stages.json').write_text(json.dumps(stages,indent=2))
   try:code=child.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    (r/(name+'_cleanup.json')).write_text(json.dumps(stop_launched_tree(child),indent=2));raise
  row.update(status='finished',exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(log))
  (r/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,str(log))
 try:
  (r/'progress.json').write_text(json.dumps(dict(stage='wait_existing_map',budget_s=1000)))
  deadline=time.monotonic()+1000
  while True:
   assert not (b/'failure.json').exists(),str(b/'failure.json')
   if (b/'result.json').exists():
    result=json.loads((b/'result.json').read_text())
    if result['status']=='DDR_OBSERVER_FULL_BOARD_MAP_NOT_RELEASE':break
   assert time.monotonic()<deadline,'existing map wait timeout';time.sleep(2)
  if a.reuse_short_batch:
   import shutil
   reused=[]
   for name,status in (('observer-mapped','DDR_OBSERVER_ACTUAL_MAP_SHORT_PASS'),('startup','FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS'),('mapped-jtag2','DDR_OBSERVER_MAPPED_JTAG_APB_PASS')):
    source=a.reuse_short_batch.resolve()/name;target=r/name
    sr=json.loads((source/'result.json').read_text());si=json.loads((source/'inputs.json').read_text())
    assert sr['status']==status and sr['inputs_sha256']==sha(source/'inputs.json')
    assert si['original_map_sha256']==sha(b/'outflow/ti60f225_oob.map.v')
    assert sha(source/'simulation.log')==sr.get('log_sha256',sr.get('simulation_sha256'))
    target.mkdir()
    for n,h in si['files'].items():assert sha(source/n)==h;shutil.copy2(source/n,target/n)
    for n in ('result.json','inputs.json','simulation.log'):shutil.copy2(source/n,target/n)
    reused.append(dict(name=name,source=str(source),result_sha256=sha(source/'result.json'),reason='Identical frozen mapped netlist; only physical seed changes'))
   (r/'reused_short.json').write_text(json.dumps(reused,indent=2))
  else:
   run('observer_mapped','validate_ddr_observer_mapped.py',['--board',b,'--root',r/'observer-mapped'],150)
   run('startup','run_ddr_observer_stage.py',['startup','--board',b,'--root',r/'startup'],180)
  physical=r/'pnr-seed1';canonical=r/'canonical-seed1';gray=r/'gray-seed1'
  run('physical','probe_control_state_ram_physical.py',[
   '--cnn-ram-reference',ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1',
   '--base',b,'--root',physical,'--pnr-budget',1500,'--logic-limit',60800,'--optimization','CONGESTION_2',
   '--fixed-dsp-layout','--seed',a.seed,'--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
   '--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],1950)
  pr=json.loads((physical/'result.json').read_text());assert all(x['exit_code']==0 for x in pr['stages'])
  import audit_control_state_implementation as impl
  impl.audit_delta=audit_delta
  (r/'implementation.json').write_text(json.dumps(impl.audit(b,physical),indent=2))
  run('canonical','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
  selected=audit(canonical);(r/'selected.json').write_text(json.dumps(selected,indent=2))
  assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
  run('gray','run_ddr_observer_stage.py',['gray','--base',physical,'--root',gray],370)
  audit_delta(b)
  result=dict(status='DDR_OBSERVER_SELECTED_ROUTE_PASS_PENDING_BUNDLE',selected=selected,gray=str(gray),
              functional_parent=str(BASE),hardware_access=False,board_download_allowed=False)
  (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as exc:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
