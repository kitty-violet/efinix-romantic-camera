"""Serial signoff reports for a completed offline route experiment."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_ddr_shift_guard_board import audit_delta,proof
from audit_canonical_multiseed import audit
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp');meta=json.loads((root/'inputs.json').read_text());board=Path(meta['board'])
 audit_delta(board);proof();r=json.loads((root/'route_result.json').read_text());physical=Path(r['physical'])
 assert r['status']=='SHIFT_GUARD_ROUTE_FINISHED_REQUIRE_STA' and physical==root/'pnr-seed1'
 assert r['route_continuation_sha256']==sha(root/'route_continuation.json')
 assert not (root/'finisher.py').exists();(root/'finisher.py').write_bytes(Path(__file__).read_bytes())
 (root/'finisher_inputs.json').write_text(json.dumps(dict(route_result_sha256=sha(root/'route_result.json'),runner_sha256=sha(__file__),board_inputs_sha256=sha(board/'inputs.json'),hardware_access=False),indent=2))
 stages=[]
 def run(name,script,args,budget):
  start=time.monotonic()
  with (root/(name+'.log')).open('x') as log:
   proc=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row=dict(stage=name,pid=proc.pid,status='running',budget_s=budget);stages.append(row);(root/'signoff_stages.json').write_text(json.dumps(stages,indent=2))
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
  row.update(status='finished',exit_code=code,seconds=time.monotonic()-start);(root/'signoff_stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,code)
 try:
  canonical=root/'canonical-seed1';gray=root/'gray-seed1'
  run('canonical','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
  selected=audit(canonical);(root/'selected.json').write_text(json.dumps(selected,indent=2))
  assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
  run('gray','audit_shift_guard_canonical_gray.py',['--base',physical,'--root',gray],370)
  result=dict(status='SHIFT_GUARD_PHYSICAL_PASS_PENDING_VENDOR_REPLAY_AND_BUNDLE',selected=selected,gray=str(gray),route_continuation_sha256=sha(root/'route_continuation.json'),finisher_inputs_sha256=sha(root/'finisher_inputs.json'),hardware_access=False,board_download_allowed=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'signoff_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
