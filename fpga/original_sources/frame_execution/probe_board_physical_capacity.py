"""Run interface and P&R only in a private copy of a measured frozen board.
No bitstream generation, programming, clock edits or timing exceptions.
"""
import argparse,json,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--pnr-budget',type=int,default=1800)
p.add_argument('--optimization',choices=['CONGESTION_1','CONGESTION_2','CONGESTION_3']);p.add_argument('--fixed-dsp-layout',action='store_true');p.add_argument('--reserve-rq-dsp',action='store_true');a=p.parse_args()
assert not (a.fixed_dsp_layout and a.reserve_rq_dsp)
assert 60<=a.pnr_budget<=3600
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
meta=json.loads((base/'inputs.json').read_text());mapped=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==mapped['inputs_sha256']
for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
assert sha(base/'outflow/ti60f225_oob.map.v')==mapped['map_sha256']
root.mkdir()
for d in ('rtl','assembly','outflow','work_syn'):
 if (base/d).exists():shutil.copytree(base/d,root/d)
for f in base.iterdir():
 if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
records=[dict(snapshot=str(root/Path(r['snapshot']).relative_to(base)),sha256=r['sha256']) for r in meta['inputs']]
for r in records:assert sha(r['snapshot'])==r['sha256']
optimization_change=None
if a.optimization:
 import re
 xml=root/'ti60f225_oob.xml';before=xml.read_text()
 after,n=re.subn(r'(name="optimization_level" value=")[^"]+(" value_type="e_option")',lambda m:m[1]+a.optimization+m[2],before)
 assert n==1
 xml.write_text(after,encoding='utf-8')
 optimization_change=dict(before_sha256=sha(base/xml.name),after_sha256=sha(xml),value=a.optimization,scope='P&R option only; same mapped netlist, RTL, SDC, pins and clocks')
 for r in records:
  if Path(r['snapshot'])==xml:r['sha256']=sha(xml)
dsp_layout=None
if a.fixed_dsp_layout or a.reserve_rq_dsp:
 from fixed_board_dsp_locations import generate
 dsp_layout=generate(root,rq_only=a.reserve_rq_dsp)
(root/'physical_inputs.json').write_text(json.dumps(dict(base=str(base),base_inputs_sha256=sha(base/'inputs.json'),base_map_sha256=mapped['map_sha256'],inputs=records,optimization_change=optimization_change,dsp_layout=dsp_layout,board_download_allowed=False,flows=['interface','pnr'],pnr_budget_seconds=a.pnr_budget,runner_sha256=sha(__file__)),indent=2))
stages=[]
try:
 for flow,budget in [('interface',300),('pnr',a.pnr_budget)]:
  script=root/(flow+'.cmd');script.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow '+flow+' --timeout '+str(budget)+'\nexit /b %errorlevel%\n')
  start=time.monotonic()
  with (root/(flow+'.log')).open('w') as log:
   proc=subprocess.Popen(['cmd.exe','/d','/c',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   try:proc.wait(timeout=budget+60)
   except subprocess.TimeoutExpired:
    # Keep the known wrapper alive until its own descendants are terminated.
    # subprocess.run would kill only cmd.exe and orphan efx_pnr on Windows.
    if proc.poll() is None:
     cleanup=stop_launched_tree(proc)
     (root/(flow+'_timeout_cleanup.json')).write_text(json.dumps(dict(wrapper_pid=proc.pid,**cleanup),indent=2))
    raise
  stages.append(dict(flow=flow,exit_code=proc.returncode,seconds=time.monotonic()-start,log_sha256=sha(root/(flow+'.log'))));(root/'stages.json').write_text(json.dumps(stages,indent=2))
  if proc.returncode:break
 for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256'],'Original candidate changed'
 assert sha(base/'outflow/ti60f225_oob.map.v')==mapped['map_sha256']
 result=dict(status='PHYSICAL_CAPACITY_PROBE_FINISHED',stages=stages,strict_sta_pass=False,board_download_allowed=False,source_unchanged=True)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),stages=stages),indent=2));raise
