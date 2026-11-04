"""Pack/place concurrently with serial same-map gates in one bounded Job."""
import concurrent.futures,json,os,subprocess,sys,time
from run_units import ROOT,HERE,sha
from snapshot_ram_board import BOARD,audit_delta
RUN=ROOT/'tmp/frame-execution/snapshot-ram-placement-checks1'
PHYSICAL=ROOT/'tmp/frame-execution/snapshot-ram-place1'
EVIDENCE=ROOT/'tmp/frame-execution/snapshot-ram-evidence1'
def main():
 audit_delta(BOARD);assert not RUN.exists();RUN.mkdir()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 commands={
  'place':[str(HERE/'place_snapshot_ram_board.py'),'--base',str(BOARD),'--root',str(PHYSICAL),
   '--logic-limit','60800','--pnr-budget','1500','--optimization','CONGESTION_2','--seed','3','--fixed-dsp-layout',
   '--cnn-ram-reference',str(ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1'),
   '--clock-profile-proof',str(ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json')],
  'checks':[str(HERE/'run_snapshot_ram_evidence.py'),'--board',str(BOARD),'--root',str(EVIDENCE)]}
 (RUN/'inputs.json').write_text(json.dumps(dict(commands=commands,inputs_sha256=sha(BOARD/'inputs.json'),
  runner_sha256=sha(__file__),sources={c[0]:sha(c[0]) for c in commands.values()},
  scope='Native placement stops for review; independent functional checks; no hardware.'),indent=2))
 def execute(name):
  start=time.monotonic()
  with (RUN/(name+'.log')).open('w',encoding='utf8') as log:
   p=subprocess.run([sys.executable,'-B',*commands[name]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
    timeout=1750 if name=='place' else 750)
  result=dict(exit_code=p.returncode,seconds=time.monotonic()-start,log_sha256=sha(RUN/(name+'.log')))
  (RUN/(name+'-result.json')).write_text(json.dumps(result,indent=2));return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs={n:pool.submit(execute,n) for n in commands};results={}
  for n,job in jobs.items():
   try:results[n]=job.result()
   except Exception as error:results[n]=dict(error=repr(error))
 passed=all(r.get('exit_code')==0 for r in results.values())
 result=dict(status='RAM_SNAPSHOT_PLACED_AND_MAPPED_SHORTS_AWAIT_REVIEW' if passed else 'INCOMPLETE',
  results=results,route_started=False,board_release=False,hardware_access=False)
 (RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
