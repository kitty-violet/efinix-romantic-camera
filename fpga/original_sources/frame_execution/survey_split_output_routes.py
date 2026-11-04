"""Compare 512 endpoint samples per frozen route in two bounded STA processes."""
import concurrent.futures,json,re,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree

BASE=ROOT/'tmp/frame-execution'
RUN=BASE/'split-outputs-route-survey1'

def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')

def analyze(folder):
 file=folder/'outflow/survey_setup.rpt';rows=[]
 for block in re.split(r'(?=^Slack\s*:)',file.read_text(),flags=re.M)[1:]:
  data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]
  nets=[(float(t),n.strip()) for t,n in re.findall(r'^\s*net \(fo=\d+\)\s+([\d.]+)\s+[\d.]+\s+(.*)',data,re.M)]
  assert nets,'Missing data arcs'
  row=dict(slack_ns=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',block)[1]),
   start=re.search(r'^Path Start\s*:\s*(.*)',block,re.M)[1].strip(),
   end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip(),
   net_ns=sum(x[0] for x in nets),max_net=max(nets))
  rows.append(row)
 assert len(rows)==512
 groups={}
 for row in rows:
  if row['slack_ns']>=0:continue
  key=re.sub(r'\[\d+\]','[]',row['start'])+' -> '+re.sub(r'\[\d+\]','[]',row['end'])
  groups.setdefault(key,[]).append(row)
 return dict(report=str(file),report_sha256=sha(file),sampled_endpoints=512,
  negative_in_sample=sum(r['slack_ns']<0 for r in rows),first_slack=rows[0]['slack_ns'],last_slack=rows[-1]['slack_ns'],
  families=[dict(path=k,count=len(v),worst=min(r['slack_ns'] for r in v),
    mean_net_ns=sum(r['net_ns'] for r in v)/len(v),worst_row=min(v,key=lambda r:r['slack_ns'])) for k,v in groups.items()],
  rows=rows,limitation='Bounded 512 worst endpoint sample, not all timing paths or a release proof.')

def worker(label,source):
 folder=RUN/label;template=HERE/'survey_routed_paths.py';code=template.read_text()
 code=code.replace('import argparse,collections,json,re,shutil,subprocess,time',
  'import argparse,collections,json,re,shutil,subprocess,time,sys\nsys.path.insert(0,'+repr(str(HERE))+')')
 assert code.count('-npaths 256')==1
 code=code.replace('-npaths 256','-npaths 512').replace("==256","==512")
 old="data=block.split('Data Path',2)[-1].split('Capture Clock Path',1)[0]"
 assert code.count(old)==1
 code=code.replace(old,"data=re.split(r'^Data Path\\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]")
 driver=RUN/(label+'-driver.py');compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 save(RUN/(label+'-driver.json'),dict(source=str(template),source_sha256=sha(template),driver_sha256=sha(driver),
  changes='512 endpoint limit and corrected data-section parsing; original geometry/constraints unchanged.'))
 start=time.monotonic()
 with (RUN/(label+'.log')).open('x') as log:
  proc=subprocess.Popen([sys.executable,'-B',str(driver),'--base',str(source),'--root',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  try:exit_code=proc.wait(timeout=250)
  except subprocess.TimeoutExpired:
   save(RUN/(label+'-timeout.json'),stop_launched_tree(proc));raise
 assert exit_code==0,(label,exit_code)
 result=analyze(folder);result['seconds']=time.monotonic()-start
 save(RUN/(label+'-analysis.json'),result)
 return result

def main():
 assert not RUN.exists()
 sources=dict(c2=BASE/'split-outputs-route1/canonical-seed3',t1=BASE/'split-outputs-timing-route1/canonical-seed3')
 for folder in sources.values():
  result=json.loads((folder/'result.json').read_text())
  assert result['status']=='CANONICAL_ROUTED_TIMING_FAIL' and result['original_and_route_unchanged']
 RUN.mkdir();save(RUN/'inputs.json',dict(sources={k:str(v) for k,v in sources.items()},
  results_sha256={k:sha(v/'result.json') for k,v in sources.items()},runner_sha256=sha(__file__),
  parallel_sta_processes=2,hardware_access=False))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures={k:pool.submit(worker,k,v) for k,v in sources.items()}
   results={k:f.result() for k,f in futures.items()}
  summary=dict(status='TWO_FROZEN_ROUTES_ENDPOINT_SURVEY_COMPLETE_NOT_RELEASE',hardware_access=False,
   results={k:{key:r[key] for key in ('sampled_endpoints','negative_in_sample','first_slack','last_slack','seconds','families')} for k,r in results.items()})
  save(RUN/'result.json',summary)
  print(json.dumps({k:{n:v[n] for n in ('negative_in_sample','first_slack','last_slack','seconds')} for k,v in results.items()}),flush=True)
 except Exception as exc:
  save(RUN/'failure.json',dict(error=repr(exc),hardware_access=False));raise

if __name__=='__main__':main()
