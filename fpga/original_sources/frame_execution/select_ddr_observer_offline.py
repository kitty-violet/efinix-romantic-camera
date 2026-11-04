"""Choose first completely passing independent route; finish offline only."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--batches',type=Path,nargs='+',required=True)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 (r/'runner.py').write_bytes(Path(__file__).read_bytes());start=time.monotonic()
 try:
  while True:
   passed=[];failed=[]
   for b in map(Path.resolve,a.batches):
    if (b/'result.json').exists():
     result=json.loads((b/'result.json').read_text())
     assert result['status']=='DDR_OBSERVER_SELECTED_ROUTE_PASS_PENDING_BUNDLE'
     passed.append(b)
    elif (b/'failure.json').exists():failed.append(str(b))
   if passed:chosen=passed[0];break
   assert len(failed)<len(a.batches),('All candidate physical gates failed',failed)
   assert time.monotonic()-start<2900,'Independent candidate wait exceeded budget'
   time.sleep(5)
  receipt=dict(chosen=str(chosen),result_sha256=sha(chosen/'result.json'),other_candidates=[str(b.resolve()) for b in a.batches if b.resolve()!=chosen],hardware_access=False)
  (r/'selection.json').write_text(json.dumps(receipt,indent=2))
  with (r/'finish.log').open('x') as log:
   child=subprocess.Popen([sys.executable,str(HERE/'finish_ddr_observer_offline.py'),'--board',str(a.board.resolve()),'--batch',str(chosen),'--root',str(r/'finish'),'--output',str(a.output.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   try:code=child.wait(timeout=750)
   except subprocess.TimeoutExpired:
    (r/'cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
  assert code==0,('Offline finish failed',str(r/'finish.log'))
  result=json.loads((r/'finish/result.json').read_text());assert result['status']=='DDR_OBSERVER_OFFLINE_READY_AWAIT_USER_SUPERVISION'
  (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as exc:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
