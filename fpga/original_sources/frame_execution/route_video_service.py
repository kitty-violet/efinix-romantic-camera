"""One reviewed, bounded route of the frozen video-service placement.

This is implementation evidence, never permission to download. The explicit
review binds current placement metrics and functional evidence. Original STA,
CDC, pin and final artifact checks remain separate mandatory release gates.
"""
import argparse,json,re,shutil,subprocess,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree

def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--budget',type=int,default=2700);a=p.parse_args()
 physical=a.physical.resolve();root=a.root.resolve();review=a.review.resolve()
 assert physical.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
 assert 60<=a.budget<=2700
 prior=json.loads((physical/'result.json').read_text());meta=json.loads((physical/'physical_inputs.json').read_text())
 assert prior['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE' and all(s['exit_code']==0 for s in prior['stages'])
 optimization=meta['optimization_change']['value']
 assert optimization in ('CONGESTION_1','CONGESTION_2','CONGESTION_3')
 recipes=[n.attrib['value'] for n in ET.parse(physical/'ti60f225_oob.xml').iter() if n.attrib.get('name')=='optimization_level']
 assert recipes==[optimization],('Placement recipe differs from route recipe',recipes,optimization)
 board=Path(meta['base']);mapped=board/'outflow/ti60f225_oob.map.v'
 assert sha(mapped)==meta['base_map_sha256']==sha(physical/'outflow/ti60f225_oob.map.v')
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 r=json.loads(review.read_text());assert r['decision']=='PROCEED_TO_BOUNDED_ROUTE' and r['physical']==str(physical)
 assert r['placement_review_sha256']==sha(physical/'placement_review_inputs.json')
 assert r['rationale'] and r['remaining_risks'] and r['required_evidence']
 for name,h in r['required_evidence'].items():assert sha(name)==h,name
 assert not (physical/'outflow/ti60f225_oob.troutingtraces').exists()
 geometry=[physical/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.vdb','.place','.out.placeloc')]
 frozen={str(f):sha(f) for f in geometry+[physical/'ti60f225_oob.sdc',physical/'ti60f225_oob.xml']}
 root.mkdir();shutil.copy2(__file__,root/'runner.py')
 manifest=dict(physical=str(physical),board=str(board),optimization=optimization,review=str(review),review_sha256=sha(review),
  required_evidence=r['required_evidence'],frozen=frozen,budget_s=a.budget,runner_sha256=sha(__file__),hardware_access=False,board_release=False)
 save(root/'inputs.json',manifest)
 before=physical/'before_route';before.mkdir()
 for f in (physical/'outflow').glob('*.place*'):shutil.copy2(f,before/f.name)
 shutil.copy2(physical/'result.json',before/'result.json')
 command=physical/'route_service.cmd';assert not command.exists()
 command.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\n'
  'call efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts route max_router_iterations=100 troute_options=max_iterations=100 optimization_level='+optimization+' --timeout '+str(a.budget)+'\nexit /b %errorlevel%\n')
 start=time.monotonic();stages=[]
 try:
  with (root/'route.log').open('x') as log:
   proc=subprocess.Popen(['cmd.exe','/d','/c',str(command)],cwd=physical,stdout=log,stderr=subprocess.STDOUT)
   stages=[dict(flow='pnr',phase='route_after_review',pid=proc.pid,status='running',budget_s=a.budget+45)]
   save(root/'stages.json',stages)
   try:rc=proc.wait(timeout=a.budget+45)
   except subprocess.TimeoutExpired:
    save(root/'timeout-stop.json',stop_launched_tree(proc));raise
  stages[0].update(status='finished',exit_code=rc,seconds=time.monotonic()-start,log_sha256=sha(root/'route.log'))
  save(root/'stages.json',stages)
  assert rc==0,('route_exit',rc)
  assert re.search(r'pnr\s*:\s*PASS',(root/'route.log').read_text())
  route=physical/'outflow/ti60f225_oob.route.out'
  iterations=re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)',route.read_text())
  assert iterations and int(iterations[-1][1])==0
  traces=physical/'outflow/ti60f225_oob.troutingtraces';assert traces.is_file()
  for f,h in frozen.items():assert sha(f)==h,('frozen_input_changed',f)
  for f,h in r['required_evidence'].items():assert sha(f)==h,('evidence_changed',f)
  assert sha(__file__)==manifest['runner_sha256']
  save(physical/'stages.json',prior['stages']+stages)
  save(physical/'result.json',dict(status='PHYSICAL_CAPACITY_PROBE_FINISHED',stages=prior['stages']+stages,
    strict_sta_pass=False,board_download_allowed=False,source_unchanged=True))
  save(root/'result.json',dict(status='VIDEO_SERVICE_LEGAL_ROUTE_PASS_NOT_RELEASE',iterations=int(iterations[-1][0]),
   trace_sha256=sha(traces),inputs_sha256=sha(root/'inputs.json'),seconds=time.monotonic()-start,
   hardware_access=False,board_release=False,original_sta_pending=True,cdc_pending=True))
 except Exception as e:
  save(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,hardware_access=False,board_release=False));raise

if __name__=='__main__':main()
