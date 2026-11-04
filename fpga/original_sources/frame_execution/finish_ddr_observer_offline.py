"""Wait for existing physical flow, finish offline bundle, stop before hardware."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
def main():
 p=argparse.ArgumentParser()
 for n in ('board','batch','root','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();batch=a.batch.resolve();r=a.root.resolve();out=a.output.resolve()
 assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir();stages=[]
 (r/'runner.py').write_bytes(Path(__file__).read_bytes())
 def wait_result(path,budget):
  start=time.monotonic()
  while not path.exists():
   for f in (batch/'failure.json',b/'failure.json',batch/'cancelled.json'):
    assert not f.exists(),str(f)
   assert time.monotonic()-start<budget,('Waiting deadline',str(path))
   time.sleep(3)
 def run(name,script,args,budget):
  start=time.monotonic();log=r/(name+'.log')
  with log.open('x') as f:
   child=subprocess.Popen([sys.executable,str(script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   row=dict(stage=name,status='running',pid=child.pid,budget_s=budget);stages.append(row)
   (r/'stages.json').write_text(json.dumps(stages,indent=2))
   try:code=child.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    (r/(name+'_cleanup.json')).write_text(json.dumps(stop_launched_tree(child),indent=2));raise
  row.update(status='finished',exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(log))
  (r/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,str(log))
 try:
  wait_result(batch/'startup/result.json',1200)
  if not (batch/'mapped-jtag2/result.json').exists():
   run('mapped_jtag',HERE/'validate_ddr_observer_mapped_jtag.py',['--startup',batch/'startup','--root',batch/'mapped-jtag2'],150)
  else:
   test=batch/'mapped-jtag2';tm=json.loads((test/'inputs.json').read_text());tr=json.loads((test/'result.json').read_text())
   assert tr['status']=='DDR_OBSERVER_MAPPED_JTAG_APB_PASS' and tr['inputs_sha256']==sha(test/'inputs.json')
   assert tm['original_map_sha256']==sha(b/'outflow/ti60f225_oob.map.v')
   for n,h in tm['files'].items():assert sha(test/n)==h
   assert tr['log_sha256']==sha(test/'simulation.log')
   (r/'reused_mapped_jtag.json').write_text(json.dumps(dict(source=str(test),result_sha256=sha(test/'result.json')),indent=2))
  wait_result(batch/'result.json',3200)
  result=json.loads((batch/'result.json').read_text());assert result['status']=='DDR_OBSERVER_SELECTED_ROUTE_PASS_PENDING_BUNDLE'
  run('bit',HERE/'prepare_routed_bitstream.py',['--canonical',result['selected']['canonical'],'--root',batch/'bit'],300)
  run('bundle',HERE/'prepare_ddr_observer_release.py',['--board',b,'--batch',batch,'--bitroot',batch/'bit','--output',out],180)
  run('preflight',out/'program.py',['--report',r/'preflight.json'],90)
  report=dict(status='DDR_OBSERVER_OFFLINE_READY_AWAIT_USER_SUPERVISION',bundle=str(out),manifest_sha256=sha(out/'manifest.json'),
              stages=stages,hardware_access=False,automatic_download_allowed=False)
  (r/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 except Exception as exc:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
