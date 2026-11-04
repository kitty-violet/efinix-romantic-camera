"""Run interface and P&R only in a private copy of a measured frozen board.
No bitstream generation, programming, clock edits or timing exceptions.
"""
import argparse,json,re,shutil,subprocess,time,traceback,sys
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree
RUNNER_CONTENT=Path(__file__).read_bytes()
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--pnr-budget',type=int,default=1800)
p.add_argument('--logic-limit',type=int,default=54720);p.add_argument('--packing-only',action='store_true')
p.add_argument('--optimization',choices=['CONGESTION_1','CONGESTION_2','CONGESTION_3']);p.add_argument('--fixed-dsp-layout',action='store_true');p.add_argument('--reserve-rq-dsp',action='store_true');p.add_argument('--clock-profile-proof',type=Path);p.add_argument('--seed',type=int)
p.add_argument('--setup-reserve-proof',type=Path,help='Tool-verified setup-only optimization margin; canonical routed STA still required')
p.add_argument('--related-gray-placement-reference',type=Path,help='Legal same-netlist placement for seven adjacent source/sync-head pairs')
p.add_argument('--related-gray-sources-only',action='store_true',help='Constrain only source FFs; leave synchronizer chain placement automatic')
a=p.parse_args()
assert 1<=a.logic_limit<=60800
assert not (a.fixed_dsp_layout and a.reserve_rq_dsp)
assert not a.related_gray_sources_only or a.related_gray_placement_reference
assert 60<=a.pnr_budget<=3600
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
meta=json.loads((base/'inputs.json').read_text());mapped=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==mapped['inputs_sha256']
for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
assert sha(base/'outflow/ti60f225_oob.map.v')==mapped['map_sha256']
root.mkdir()
(root/'physical_runner.py').write_bytes(RUNNER_CONTENT)
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
seed_change=None
if a.seed is not None:
 xml=root/'ti60f225_oob.xml';before=xml.read_text();before_seed_sha=sha(xml)
 after,n=re.subn(r'(name="seed" value=")[^"]+(" value_type="e_integer")',lambda m:m[1]+str(a.seed)+m[2],before)
 assert n==1
 xml.write_text(after,encoding='utf-8')
 seed_change=dict(seed=a.seed,before_sha256=before_seed_sha,after_sha256=sha(xml),scope='Official place-and-route seed only; mapped RTL and all timing requirements unchanged')
 for r in records:
  if Path(r['snapshot'])==xml:r['sha256']=sha(xml)
clock_profile=None
if a.clock_profile_proof:
 from measured_pinmap_clock_profile import apply_profile
 clock_profile=apply_profile(root,a.clock_profile_proof.resolve())
 for r in records:
  if Path(r['snapshot'])==root/'ti60f225_oob.sdc':r['sha256']=sha(root/'ti60f225_oob.sdc')
setup_margin=None
if a.setup_reserve_proof:
 assert a.clock_profile_proof,'Apply only after the audited PLL relationship'
 proof_path=a.setup_reserve_proof.resolve();proof=json.loads(proof_path.read_text())
 assert proof['status']=='EXTRA_SETUP_MARGIN_SAME_ROUTE_VERIFIED'
 zero,reserved=proof['rows'];reserve=reserved['reserve_ns']
 assert zero['reserve_ns']==0 and 0.3<=reserve<=1.0 and abs(reserve-round(reserve,3))<1e-9
 assert abs(zero['setup']['slack_ns']-reserved['setup']['slack_ns']-reserve)<0.0011
 assert zero['hold']['slack_ns']==reserved['hold']['slack_ns']
 for row in proof['rows']:
  for kind in ('setup','hold'):assert sha(row[kind]['report'])==row[kind]['sha256']
 sdc=root/'ti60f225_oob.sdc';before=sdc.read_text()
 assert not re.search(r'^\s*set_clock_uncertainty\b',before,re.M)
 canonical=root/'canonical_timing.sdc';shutil.copy2(sdc,canonical)
 extra='\n# Reserve '+f'{reserve:.3f}'+'ns during optimization; final acceptance is unchanged.\nset_clock_uncertainty -setup -to [get_clocks {*}] '+f'{reserve:.3f}'+'\n'
 sdc.write_text(before+extra,encoding='utf-8')
 setup_margin=dict(extra_setup_ns=reserve,extra_hold_ns=0,proof=str(proof_path),proof_sha256=sha(proof_path),canonical_sdc=str(canonical),canonical_sdc_sha256=sha(canonical),optimization_sdc_sha256=sha(sdc),canonical_final_sta_required=True)
 for r in records:
  if Path(r['snapshot'])==sdc:r['sha256']=sha(sdc)
dsp_layout=None
if a.fixed_dsp_layout or a.reserve_rq_dsp:
 from fixed_board_dsp_locations import generate
 dsp_layout=generate(root,rq_only=a.reserve_rq_dsp)
related_gray_layout=None
if a.related_gray_placement_reference:
 assert a.fixed_dsp_layout
 from tight_related_gray_locations import generate as tighten_gray
 related_gray_layout=tighten_gray(root,a.related_gray_placement_reference,sources_only=a.related_gray_sources_only)
 dsp_layout['assignments_sha256']=sha(root/'fixed_dsp.placeloc')
(root/'physical_inputs.json').write_text(json.dumps(dict(base=str(base),base_inputs_sha256=sha(base/'inputs.json'),base_map_sha256=mapped['map_sha256'],inputs=records,optimization_change=optimization_change,seed_change=seed_change,clock_profile=clock_profile,setup_margin=setup_margin,dsp_layout=dsp_layout,related_gray_layout=related_gray_layout,board_download_allowed=False,flows=['interface','pnr'],pnr_budget_seconds=a.pnr_budget,logic_limit=a.logic_limit,runner_sha256=sha(__file__)),indent=2))
stages=[]
try:
 for flow,budget in [('interface',300),('pnr',a.pnr_budget)]:
  script=root/(flow+'.cmd');script.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow '+flow+' --timeout '+str(budget)+'\nexit /b %errorlevel%\n')
  capacity_file=root/'outflow/ti60f225_oob.place.rpt'
  if flow=='pnr':assert not capacity_file.exists(),'Reject stale packing report'
  capacity_checked=False
  start=time.monotonic()
  with (root/(flow+'.log')).open('w') as log:
   proc=subprocess.Popen(['cmd.exe','/d','/c',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   try:
    while proc.poll() is None:
     if time.monotonic()-start>budget+60:
      raise subprocess.TimeoutExpired(proc.args,budget+60)
     if flow=='pnr' and not capacity_checked and capacity_file.exists():
      contents=capacity_file.read_text(errors='replace')
      found=re.search(r'^XLRs:\s*(\d+)\s*/\s*60800',contents,re.M)
      if found:
       used=int(found[1]);capacity_checked=True
       gate=dict(status='INITIAL_PACKING_WITHIN_LIMIT_NOT_RELEASE' if used<=a.logic_limit else 'PACKING_EXCEEDS_PROJECT_LIMIT',
        used_xlr=used,limit_xlr=a.logic_limit,seconds=time.monotonic()-start,
        place_report_sha256=sha(capacity_file),map_sha256=mapped['map_sha256'],board_release=False)
       if used>a.logic_limit:
        gate['cleanup']=stop_launched_tree(proc)
       (root/'capacity_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
       if used>a.logic_limit:
        raise RuntimeError(f'Packing {used} exceeds project limit {a.logic_limit}; own P&R tree stopped')
       if a.packing_only:
        gate['cleanup']=stop_launched_tree(proc)
        (root/'capacity_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
        for rec in meta['inputs']:assert sha(rec['snapshot'])==rec['sha256']
        (root/'result.json').write_text(json.dumps(dict(status='PACKING_ONLY_PROBE_FINISHED',used_xlr=used,strict_sta_pass=False,board_download_allowed=False),indent=2)+'\n')
        print(json.dumps(dict(status='PACKING_ONLY_PROBE_FINISHED',used_xlr=used)),flush=True)
        sys.exit(0)
     time.sleep(1)
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
