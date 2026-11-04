"""Serial corrected primitive tests and full-board freeze/map; fail immediately."""
import json,subprocess,sys,time
from run_units import ROOT,HERE,sha
from snapshot_ram_board import BOARD,SHORT,BRIDGE
RUN=ROOT/'tmp/frame-execution/snapshot-ram-preflight2'
def main():
 assert not RUN.exists();RUN.mkdir();stages=[]
 commands=[('short','run_snapshot_ram_shorts.py',['--root',str(SHORT)],350),
  ('bridge','run_snapshot_ram_bridge.py',['--root',str(BRIDGE)],180),
  ('freeze','prepare_snapshot_ram_board.py',[],80),
  ('map','map_prepared_board.py',['--root',str(BOARD)],1000)]
 for name,script,args,budget in commands:
  start=time.monotonic();row=dict(name=name,status='RUNNING',budget=budget);stages.append(row)
  (RUN/'stages.json').write_text(json.dumps(stages,indent=2))
  with (RUN/(name+'.log')).open('x',encoding='utf8') as f:
   p=subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=budget)
  row.update(status='PASS' if p.returncode==0 else 'FAIL',exit_code=p.returncode,seconds=time.monotonic()-start,
   log_sha256=sha(RUN/(name+'.log')))
  (RUN/'stages.json').write_text(json.dumps(stages,indent=2))
  if p.returncode:
   (RUN/'failure.json').write_text(json.dumps(row,indent=2));return 1
 result=dict(status='RAM_SNAPSHOT_PRIMITIVE_BRIDGE_FULL_MAP_PASS',stages=stages,board=str(BOARD),
  inputs_sha256=sha(BOARD/'inputs.json'),map_result_sha256=sha(BOARD/'result.json'),hardware_access=False)
 (RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
