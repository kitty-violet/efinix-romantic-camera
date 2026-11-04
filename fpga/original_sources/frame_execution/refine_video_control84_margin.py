"""One timing-aware C2 detail-placement refinement of the checked84 design.

No synthesis, ordinary-cell anchors, seed sweep or board access. Extra setup
pressure guides placement and route; acceptance uses the original constraints.
"""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree
from video_control84_physical import audit,verify_placed,BOARD

B=ROOT/'tmp/frame-execution'
BASE=B/'video-control84-place1'
DEST=B/'video-control84-refine3-c2margin'
PROOF=B/'video-control84-margin-proof1/result.json'

def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def prepare():
 assert not DEST.exists();audit()
 proof=json.loads(PROOF.read_text());assert proof['status']=='EXTRA_SETUP_MARGIN_SAME_ROUTE_VERIFIED'
 zero,extra=proof['rows'];assert (zero['setup']['slack_ns'],extra['setup']['slack_ns'])==(-.032,-.332)
 assert zero['hold']['slack_ns']==extra['hold']['slack_ns']==.026
 for f,h in proof['unchanged_original'].items():assert sha(f)==h
 for row in proof['rows']:
  for kind in ('setup','hold'):assert sha(row[kind]['report'])==row[kind]['sha256']
 meta=json.loads((BASE/'physical_inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 assert meta['base_map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
 assert meta['optimization_change']['value']=='CONGESTION_2' and meta['seed_change']['seed']==1
 assert sha(BASE/'ti60f225_oob.sdc')=='464ef739d92e3fb1d2b15018863be99636efc1cae42ca581a0526f305c4be262'
 # Keep packed/global layout, but never copy stale route or review verdicts.
 shutil.copytree(BASE,DEST,ignore=shutil.ignore_patterns('before_route','*.rpt','*.log','*.troutingtraces','*.route',
  '*.cmd','*.out','result.json','stages.json','failure.json','route-review.json','global-bandwidth-review.json',
  'control84-path-review.json','placement_review_inputs.json','hard-anchor-check.json',
  'original-gray-endpoints.json','original-periphery-output-check.json'))
 sdc=DEST/'ti60f225_oob.sdc';original=sdc.read_text()
 assert not re.search(r'^\s*set_clock_uncertainty\b',original,re.M)
 sdc.write_text(original+'\n# Optimization reserve only; release uses original SDC.\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\n',encoding='utf8')
 for row in meta['inputs']:
  row['snapshot']=str(DEST/Path(row['snapshot']).relative_to(BASE))
  if Path(row['snapshot'])==sdc:row['sha256']=sha(sdc)
  else:assert sha(row['snapshot'])==row['sha256']
 meta['setup_margin']=dict(extra_setup_ns=.3,extra_hold_ns=0,canonical_sdc=str(BASE/'ti60f225_oob.sdc'),
  canonical_sdc_sha256=sha(BASE/'ti60f225_oob.sdc'),optimization_sdc_sha256=sha(sdc),proof=str(PROOF),proof_sha256=sha(PROOF))
 refs=[Path(__file__).resolve(),PROOF,BASE/'physical_inputs.json',BASE/'result.json',BASE/'outflow/ti60f225_oob.place',
  BASE/'outflow/ti60f225_oob.map.v',BASE/'outflow/ti60f225_oob.vdb',BASE/'fixed_dsp.placeloc',BASE/'ti60f225_oob.sdc',
  B/'video-control84-canonical1/complete-path-diagnosis.json',B/'video-control84-canonical2-margin/complete-path-diagnosis.json',
  B/'video-control86-canonical1/complete-path-diagnosis.json',
  ROOT/'tmp/board-test/video-gain82-offline1/efinity-timing-closure-placement-excerpts.txt']
 info=dict(status='CONTROL84_C2_SETUP_AWARE_DETAIL_PLACEMENT_PREPARED',source=str(BASE),optimization='CONGESTION_2',seed=1,
  tool_budget_s=900,host_budget_s=945,hard_anchors=404,ordinary_anchors=0,old_route_imported=False,
  source_map_and_packing_unchanged=True,extra_setup_ns=.3,extra_hold_ns=0,
  rationale=[
   'Measured86 full2000 endpoint survey has966 failures across the design. Do not chase isolated high-fanout nets on that globally worse implementation.',
   'Measured84 original has2 negative endpoints on the same15-level expect_bank-to-rgb_head chain, WNS-.032ns. Same-placement setup-pressure reroute left one-.023ns miss. Its geometry must now be improved before route.',
   'Retain84 bandwidth fixes and checked mapping, original C2 seed1 global layout and404hard anchors. Run official detailed_place with the same C2 recipe and validated+.300ns setup pressure, all ordinary cells free.',
   'This is a distinct placement correction, not another identical reroute. Stop for all2000 old-path geometry, congestion, reset/clock and interface review before routing.',
   'Never infer route success from geometry. Independently restore the byte-exact original SDC for canonical WNS/WHS, Gray and PLL gates; no clock frequency/pin/reset or exception change.'
  ],evidence={str(p):sha(p) for p in refs},hardware_access=False,board_release=False)
 meta['placement_refinement']=info;save(DEST/'physical_inputs.json',meta);save(DEST/'refinement-inputs.json',info)
 print(json.dumps(dict(status=info['status'],root=str(DEST)),ensure_ascii=False),flush=True)

def run():
 info=json.loads((DEST/'refinement-inputs.json').read_text());assert not (DEST/'running.json').exists()
 for f,h in info['evidence'].items():assert sha(f)==h,f
 frozen={str(DEST/n):sha(DEST/n) for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb',
  'outflow/ti60f225_oob.clstdb','outflow/ti60f225_oob.cnst','ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.xml')}
 cmd=DEST/'refine.cmd';cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts detailed_place optimization_level=CONGESTION_2 --timeout 900\nexit /b %errorlevel%\n')
 start=time.monotonic()
 try:
  with (DEST/'refine.log').open('x') as log:
   proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=DEST,stdout=log,stderr=subprocess.STDOUT)
   save(DEST/'running.json',dict(pid=proc.pid,budget_s=945,hardware_access=False))
   try:rc=proc.wait(timeout=945)
   except subprocess.TimeoutExpired:
    save(DEST/'timeout-stop.json',stop_launched_tree(proc));raise
  assert rc==0,rc
  assert re.search(r'pnr\s*:\s*PASS',(DEST/'refine.log').read_text())
  for f,h in frozen.items():assert sha(f)==h,f
  for f,h in info['evidence'].items():assert sha(f)==h,f
  assert sha(DEST/'outflow/ti60f225_oob.place')!=sha(BASE/'outflow/ti60f225_oob.place')
  assert not (DEST/'outflow/ti60f225_oob.troutingtraces').exists()
  save(DEST/'hard-anchor-check.json',verify_placed(DEST))
  result=dict(status='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE',seconds=time.monotonic()-start,
   frozen=frozen,inputs_sha256=sha(DEST/'refinement-inputs.json'),hardware_access=False,board_release=False)
  save(DEST/'refinement-result.json',result)
  stages=[dict(s,reused_from=str(BASE),not_reexecuted=True) for s in json.loads((BASE/'before_route/result.json').read_text())['stages']]
  stages.append(dict(flow='detailed_place',phase='C2_SETUP_MARGIN_0p300',exit_code=rc,seconds=result['seconds'],log_sha256=sha(DEST/'refine.log')))
  save(DEST/'stages.json',stages)
  save(DEST/'result.json',dict(status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE',stages=stages,source_unchanged=True,strict_sta_pass=False,board_download_allowed=False))
  print(json.dumps(result),flush=True)
 except Exception as exc:
  save(DEST/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 else:run()
