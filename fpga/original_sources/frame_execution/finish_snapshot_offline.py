"""One bounded offline finish after four seeds; never calls USB/programming."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path);p.add_argument('--batch',required=True,type=Path)
 p.add_argument('--reuse-gray',type=Path)
 a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve()
 done=json.loads((batch/'result.json').read_text());assert done['status']=='SNAPSHOT_FOUR_SEED_BASIC_TIMING_PASS_PENDING_CDC'
 selected=done['selected'];physical=Path(selected['physical']);canonical=Path(selected['canonical'])
 out=batch/'offline-finish';assert not out.exists();out.mkdir()
 stages=[]
 def run(name,args,budget,cwd=ROOT):
  started=time.monotonic()
  with (out/(name+'.log')).open('w') as log:
   proc=subprocess.Popen([sys.executable,*map(str,args)],cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
  row=dict(stage=name,exit_code=code,seconds=time.monotonic()-started,log_sha256=sha(out/(name+'.log')))
  stages.append(row);assert code==0,row
  return row
 gray=batch/f'gray-seed{selected["seed"]}';bit=batch/f'bit-seed{selected["seed"]}'
 if a.reuse_gray:
  gray=a.reuse_gray.resolve()
  assert json.loads((gray/'result.json').read_text())['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
  assert Path(json.loads((gray/'inputs.json').read_text())['base'])==physical
 run('implementation',[HERE/'audit_snapshot_implementation.py','--board',board,'--physical',physical,'--out',batch/'snapshot_implementation.json'],60)
 run('host_tests',['-m','unittest','test_control_snapshot_host','-v'],30,HERE)
 (batch/'host_tests.log').write_bytes((out/'host_tests.log').read_bytes())
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(run,'bit',[HERE/'prepare_routed_bitstream.py','--canonical',canonical,'--root',bit],300)]
  if not a.reuse_gray:jobs.append(pool.submit(run,'gray',[HERE/'audit_snapshot_routed_gray.py','--base',physical,'--root',gray],370))
  for job in jobs:job.result()
 review=batch/'snapshot_cdc_review.json'
 run('review',[HERE/'review_snapshot_cdc.py','--board',board,'--batch',batch,'--gray',gray,'--output',review],90)
 output=ROOT/'output/deployment/board-control-snapshot-20261009'
 run('bundle',[HERE/'prepare_snapshot_release.py','--board',board,'--batch',batch,'--gray',gray,'--bitroot',bit,'--review',review,'--output',output],100)
 # Default programmer mode performs evidence/hash checks only, no connection.
 run('offline_validate',[output/'program.py','--report',out/'offline-validation.json'],60)
 result=dict(status='OFFLINE_SNAPSHOT_READY_AWAIT_SUPERVISION',stages=stages,bundle=str(output),
  manifest_sha256=sha(output/'manifest.json'),hardware_access=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
