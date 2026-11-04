"""Run the second actual allocation partition through the exact frozen binary."""
import argparse,json,os,re,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import run,BIN,VROOT
def main():
 p=argparse.ArgumentParser()
 for k in ('root','native','vectors-source'):p.add_argument('--'+k,required=True,type=Path)
 a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  native=json.loads((a.native/'inputs.json').read_text());vectors=json.loads((a.vectors_source/'inputs.json').read_text())
  binary=a.native/'obj/Vtb_residual_column.exe';assert binary.is_file()
  for m in (native,vectors):
   for r in m['inputs']:assert sha(r['snapshot'])==r['sha256']
  nr={Path(r['snapshot']).name:r['sha256'] for r in native['inputs']}
  for r in vectors['inputs']:assert nr[Path(r['snapshot']).name]==r['sha256']
  for name,h in vectors['vectors'].items():assert sha(a.vectors_source/'vectors'/name)==h
  manifest=dict(native_manifest=str(a.native/'inputs.json'),native_manifest_sha256=sha(a.native/'inputs.json'),
   binary=str(binary),binary_sha256=sha(binary),source_manifest_sha256=sha(a.vectors_source/'inputs.json'),oracle=vectors['oracle'],
   vectors=vectors['vectors'],runner_sha256=sha(__file__))
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],VERILATOR_ROOT=str(VROOT))
  body,seconds=run([binary.resolve(),'+VECTORS='+(a.vectors_source/'vectors').resolve().as_posix()],root,root/'simulation.log',env)
  assert not re.search(r'\bFAIL\b|%Error',body)
  passed=re.findall(r'^(PASS RESIDUAL_COLUMN.*)$',body,re.M);rows=re.findall(r'^RESCOL (.*)$',body,re.M)
  assert len(passed)==1 and len(rows)==vectors['oracle']['cases']*2+1
  costs={}
  for row in rows:
   d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',row)}
   if d['random']==0:
    if d['case'] in costs:assert costs[d['case']]==d['cycles']
    costs[d['case']]=d['cycles']
  assert len(costs)==vectors['oracle']['cases']
  assert sha(binary)==manifest['binary_sha256'] and sha(a.native/'inputs.json')==manifest['native_manifest_sha256']
  result=dict(status='LOCAL_ACTUAL_RESIDUAL_PARTITION_PASS',line=passed[0],rows=rows,cycles=sum(costs.values()),seconds=seconds,
   inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),part=vectors['oracle']['part'],full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
