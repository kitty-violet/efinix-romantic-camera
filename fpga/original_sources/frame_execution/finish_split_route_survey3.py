"""Reuse the completed C2 report; run only T1's main endpoint query.

Two simultaneous vendor path enumerations exceed six GiB. Do not rerun the
successful C2 work, and do not include redundant per-source subqueries.
"""
import json,subprocess,sys,time
from pathlib import Path
from run_units import HERE,ROOT,sha
import survey_split_output_routes as survey
from windows_process_tree import stop_launched_tree

RUN=ROOT/'tmp/frame-execution/split-outputs-route-survey3'
C2=ROOT/'tmp/frame-execution/split-outputs-route-survey2/c2'
SOURCE=ROOT/'tmp/frame-execution/split-outputs-timing-route1/canonical-seed3'

def main():
 assert not RUN.exists()
 c2=survey.analyze(C2)
 inputs=json.loads((C2/'inputs.json').read_text())
 for file,digest in inputs['frozen'].items():assert sha(file)==digest
 c2_sha=c2['report_sha256']
 RUN.mkdir()
 code=(HERE/'survey_routed_paths.py').read_text()
 code=code.replace('import argparse,collections,json,re,shutil,subprocess,time',
  'import argparse,collections,json,re,shutil,subprocess,time,sys\nsys.path.insert(0,'+repr(str(HERE))+')')
 code=code.replace('-npaths 256','-npaths 512').replace('==256','==512')
 begin=code.index('    for name,pattern in ')
 end=code.index("    (root/'survey.tcl')",begin)
 code=code[:begin]+code[end:]
 before="data=block.split('Data Path',2)[-1].split('Capture Clock Path',1)[0]"
 assert code.count(before)==1
 code=code.replace(before,"data=re.split(r'^Data Path\\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]")
 driver=RUN/'t1-driver.py';compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 survey.save(RUN/'inputs.json',dict(c2_reused_report=c2['report'],c2_reused_sha256=c2_sha,
  source=str(SOURCE),source_result_sha256=sha(SOURCE/'result.json'),driver_sha256=sha(driver),
  generator_sha256=sha(__file__),parallel_sta_processes=1,hardware_access=False))
 start=time.monotonic()
 with (RUN/'t1.log').open('x') as log:
  proc=subprocess.Popen([sys.executable,'-B',str(driver),'--base',str(SOURCE),'--root',str(RUN/'t1')],
   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=250)
  except subprocess.TimeoutExpired:
   survey.save(RUN/'timeout.json',stop_launched_tree(proc));raise
 assert rc==0,rc
 t1=survey.analyze(RUN/'t1')
 assert sha(c2['report'])==c2_sha
 for file,digest in inputs['frozen'].items():assert sha(file)==digest
 result=dict(status='TWO_FROZEN_ROUTES_ENDPOINT_SURVEY_COMPLETE_NOT_RELEASE',
  seconds=time.monotonic()-start,hardware_access=False,c2=c2,t1=t1)
 survey.save(RUN/'result.json',result)
 print(json.dumps({k:{n:r[n] for n in ('negative_in_sample','first_slack','last_slack')} for k,r in [('c2',c2),('t1',t1)]}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if RUN.exists():survey.save(RUN/'failure.json',dict(error=repr(exc)))
  raise
