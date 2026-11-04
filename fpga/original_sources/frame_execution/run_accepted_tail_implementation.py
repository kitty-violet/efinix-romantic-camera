"""Once map passes, parallel placement and serialized single-license short tests."""
import argparse,concurrent.futures,json,subprocess,sys,threading,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from accepted_tail_board import audit_delta
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();audit_delta(board)
 mapped=json.loads((board/'result.json').read_text());assert mapped['status']=='CONNECTED_BOARD_MAP_MEASURED_NOT_RELEASE'
 assert mapped['inputs_sha256']==sha(board/'inputs.json') and mapped['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 stages=[];lock=threading.Lock();stem='accepted-tail'
 physical=ROOT/'tmp/frame-execution'/f'{stem}-place1';shorts=ROOT/'tmp/frame-execution'/f'{stem}-mapped1'
 flow=HERE/'accepted_tail_flow.py'
 def save():
  with lock:(root/'stages.json').write_text(json.dumps(stages,indent=2))
 def run(name,args,budget):
  row=dict(name=name,status='running',budget_s=budget);stages.append(row);start=time.monotonic()
  with (root/(name+'.log')).open('x') as log:
   proc=subprocess.Popen([sys.executable,'-B',str(flow),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row['pid']=proc.pid;save()
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    row.update(status='timeout',cleanup=stop_launched_tree(proc));save();raise
  row.update(status='pass' if code==0 else 'failed',exit_code=code,seconds=time.monotonic()-start);save()
  assert code==0,(name,code)
 def mapped_short():
  run('audit',['audit',board],100)
  run('selector',['selector-audit',board],100)
  run('equivalence',['equivalence','--board',board,'--root',ROOT/'tmp/frame-execution/accepted-tail-control-equivalence1'],120)
  run('ofp3',['rtl','--board',board,'--root',ROOT/'tmp/frame-execution/accepted-tail-ofp3-rtl1'],120)
  run('startup',['startup','--board',board,'--root',shorts/'startup'],180)
  run('observe',['observe','--source',shorts/'startup','--root',shorts/'observe'],160)
  run('serial',['serial','--board',board,'--root',ROOT/'tmp/frame-execution/accepted-tail-mapped1-serial','--startup',shorts/'startup'],480)
 def place():
  run('place',['place','--base',board,'--root',physical,'--logic-limit',60800,'--pnr-budget',1800,
   '--optimization','CONGESTION_2','--seed',3,'--fixed-dsp-layout','--cnn-ram-reference',ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1',
   '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'],1950)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(mapped_short),pool.submit(place)]
   errors=[]
   for f in futures:
    try:f.result()
    except Exception as exc:errors.append(repr(exc))
  assert not errors,errors
  audit_delta(board)
  result=dict(status='ACCEPTED_TAIL_PLACED_AND_MAPPED_SHORTS_PASS_AWAIT_ROUTE_REVIEW',board=str(board),physical=str(physical),
   board_inputs_sha256=sha(board/'inputs.json'),stages=stages,hardware_access=False,board_release=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
 except Exception as exc:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
