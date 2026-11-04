"""Isolate vendor STA diagnostics on immutable geometry; never a release gate."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree
from survey_camera_near_critical import parse

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['survey','checks'],required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=ROOT/'tmp/frame-execution/cnn-observer-trim-free-route1/canonical-seed3';out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 failure=json.loads((base/'failure.json').read_text());assert 'EXCEPTION_ACCESS_VIOLATION' in failure['error']
 inputs=json.loads((base/'inputs.json').read_text())
 for f,h in inputs['original'].items():assert sha(f)==h,f
 files=[base/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]
 files += [base/'ti60f225_oob.sdc',base/'inputs.json',base/'failure.json',base/'outflow/canonical_setup.rpt',Path(__file__)]
 frozen={str(f):sha(f) for f in files};out.mkdir()
 for folder in ('outflow','work_pnr'):shutil.copytree(base/folder,out/folder)
 for f in base.iterdir():
  if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,out/f.name)
 command=('report_timing -setup -npaths 1000 -nworst 1 -file isolated_setup.rpt' if a.mode=='survey' else 'check_timing -file isolated_checks.rpt -verbose')
 (out/'isolated.tcl').write_text('report_timing_summary -file isolated_summary.rpt\n'+command+'\nputs {ISOLATED_COMPLETE}; flush stdout\n')
 (out/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script isolated.tcl --timeout 180\nexit /b %errorlevel%\n')
 (out/'inputs.json').write_text(json.dumps(dict(mode=a.mode,frozen=frozen,tcl_sha256=sha(out/'isolated.tcl'),release_check_bypassed=False,hardware_access=False),indent=2))
 try:
  start=time.monotonic()
  with (out/'isolated.log').open('x') as log:
   child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=out,stdout=log,stderr=subprocess.STDOUT)
   try:rc=child.wait(timeout=210)
   except subprocess.TimeoutExpired:
    (out/'timeout.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
  log=(out/'isolated.log').read_text(errors='replace')+(out/'outflow/ti60f225_oob.sta.out').read_text(errors='replace')
  assert rc==0 and 'ISOLATED_COMPLETE' in log,(rc,log[-1500:])
  assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|^\s*(?:ERROR|FATAL)\s*:',log,re.M)
  for f,h in frozen.items():assert sha(f)==h,f
  for f in files[:4]:assert sha(out/'outflow'/f.name)==frozen[str(f)]
  result=dict(status='ISOLATED_DIAGNOSTIC_COMPLETE_NOT_RELEASE',mode=a.mode,seconds=time.monotonic()-start,
   inputs_sha256=sha(out/'inputs.json'),original_and_route_unchanged=True,hardware_access=False,board_release=False)
  if a.mode=='survey':
   report=out/'outflow/isolated_setup.rpt';rows=parse(report.read_text());before=parse((base/'outflow/canonical_setup.rpt').read_text())
   assert len(rows)==1000
   for x,y in zip(before,rows):assert x==y
   result.update(paths=len(rows),worst_ns=rows[0]['slack_ns'],last_ns=rows[-1]['slack_ns'],negative_paths=sum(x['slack_ns']<0 for x in rows))
  else:
   report=out/'outflow/isolated_checks.rpt';body=report.read_text();raw={k:int(v) for k,v in re.findall(r'checking (\w+) \((\d+)\)',body)}
   assert set(raw)=={'no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','multiple_clock'}
   actual=dict(raw)
   for direction in ('input','output'):
    m=re.search(r'There are (\d+) '+direction+r' ports with no '+direction+r' delay specified',body);assert m;actual['no_'+direction+'_delay']=int(m[1])
   result.update(check_timing_raw=raw,check_timing_actual=actual)
  result.update(report=str(report),report_sha256=sha(report));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
