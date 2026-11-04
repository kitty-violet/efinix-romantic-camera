"""Congestion-aware detailed placement after the measured83 timing failure."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree
from audit_canonical_multiseed import audit as timing_audit
from video_bandwidth83_physical import audit,verify_placed

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 base=ROOT/'tmp/frame-execution/video-bandwidth83-place1';timing=ROOT/'tmp/frame-execution/video-bandwidth83-canonical1'
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 audit();t=timing_audit(timing);assert (t['wns_ns'],t['whs_ns'])==(-.182,.027) and not t['current_timing_pass']
 assert Path(t['physical'])==base
 shutil.copytree(base,root,ignore=shutil.ignore_patterns('before_route','*.rpt','*.log','*.troutingtraces','*.route','*.bit',
  '*.cmd','*.out','result.json','stages.json','failure.json','route-review.json','global-bandwidth-review.json',
  'placement_review_inputs.json','hard-anchor-check.json','original-gray-endpoints.json','original-periphery-output-check.json'))
 meta=json.loads((base/'physical_inputs.json').read_text());old_xml=sha(root/'ti60f225_oob.xml')
 xml=root/'ti60f225_oob.xml';text=xml.read_text();needle='name="optimization_level" value="CONGESTION_2"';assert text.count(needle)==1
 xml.write_text(text.replace(needle,'name="optimization_level" value="CONGESTION_1"'),encoding='utf8')
 for row in meta['inputs']:
  row['snapshot']=str(root/Path(row['snapshot']).relative_to(base))
  if Path(row['snapshot'])==xml:row['sha256']=sha(xml)
  else:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 meta['optimization_change']=dict(value='CONGESTION_1',before_sha256=old_xml,after_sha256=sha(xml),
  scope='C2 global placement reused; C1 detailed placement and subsequent route. No seed/RTL/clock/constraint change.')
 refs=[Path(__file__).resolve(),base/'physical_inputs.json',base/'result.json',base/'outflow/ti60f225_oob.place',
  base/'outflow/ti60f225_oob.map.v',base/'outflow/ti60f225_oob.vdb',base/'fixed_dsp.placeloc',base/'ti60f225_oob.sdc',
  timing/'result.json',timing/'setup-analysis.json',timing/'outflow/canonical_setup.rpt',
  ROOT/'tmp/board-test/video-gain82-offline1/efinity-timing-closure-placement-excerpts.txt',
  ROOT/'tmp/frame-execution/video-sensor73-route4-refined/failure.json']
 proof=dict(status='BANDWIDTH83_DETAILED_PLACEMENT_REVIEWED',source=str(base),optimization='CONGESTION_1',global_optimization='CONGESTION_2',
  budget_seconds=900,hard_anchors=404,ordinary_anchors=0,old_route_imported=False,
  rationale=[
   'Current83 legal route has22 negative setup endpoints:19 from expect_bank1 to coefficient-store controls, one arenaRAM-to-input-history, two APB-to-loader. Hold+.027ns and all constraint completeness checks pass.',
   'Worst14-level path is wire dominated: endpoint decode(209,31)->stop(129,75)->DDRcontrol(77,72)->DDRcontrol(77,126)->write response(198,36), then storeRAM(147,122). The arena path has2.243ns first net from(18,162) to(2,297).',
   'Use official detailed placer over the existing C2 global placement, all ordinary cells free, C1 congestion-aware timing recipe. Do not freeze all cells or hand-move a few gates.',
   'Official timing-closure guide page91 recommends congestion recipes for hard-to-route designs;83 took66 routing iterations. Prior73 TIMING_1 increased congestion then failed, so do not repeat a pure-timing recipe by default.',
   'Stop before routing. Compare all2000 current failing-implementation paths, whole-chip congestion, high-fanout controls and hard anchors; improvement at one endpoint alone is insufficient.'
  ],evidence={str(f):sha(f) for f in refs},hardware_access=False,board_release=False)
 meta['placement_refinement']=proof
 (root/'physical_inputs.json').write_text(json.dumps(meta,indent=2));(root/'refinement-inputs.json').write_text(json.dumps(proof,indent=2))
 frozen={str(root/n):sha(root/n) for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.xml')}
 cmd=root/'refine.cmd';cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts detailed_place optimization_level=CONGESTION_1 --timeout 900\nexit /b %errorlevel%\n')
 start=time.monotonic()
 try:
  with (root/'refine.log').open('x') as f:
   proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=root,stdout=f,stderr=subprocess.STDOUT)
   (root/'running.json').write_text(json.dumps(dict(pid=proc.pid,budget_s=945,hardware_access=False)))
   try:rc=proc.wait(timeout=945)
   except subprocess.TimeoutExpired:
    (root/'timeout-stop.json').write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
  assert rc==0,rc
  assert re.search(r'pnr\s*:\s*PASS',(root/'refine.log').read_text())
  for f,h in frozen.items():assert sha(f)==h,f
  for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
  for f,h in proof['evidence'].items():assert sha(f)==h,f
  assert sha(root/'outflow/ti60f225_oob.place')!=sha(base/'outflow/ti60f225_oob.place')
  assert not (root/'outflow/ti60f225_oob.troutingtraces').exists()
  anchors=verify_placed(root);(root/'hard-anchor-check.json').write_text(json.dumps(anchors,indent=2))
  result=dict(status='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE',exit_code=rc,seconds=time.monotonic()-start,inputs_sha256=sha(root/'refinement-inputs.json'),frozen=frozen,hardware_access=False,board_release=False)
  (root/'refinement-result.json').write_text(json.dumps(result,indent=2))
  stages=[dict(s,reused_from=str(base),not_reexecuted=True) for s in json.loads((base/'before_route/result.json').read_text())['stages']]
  stages.append(dict(flow='detailed_place',phase='CONGESTION_1',exit_code=rc,seconds=result['seconds'],log_sha256=sha(root/'refine.log')))
  (root/'stages.json').write_text(json.dumps(stages,indent=2))
  (root/'result.json').write_text(json.dumps(dict(status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE',stages=stages,source_unchanged=True,strict_sta_pass=False,board_download_allowed=False),indent=2))
  print(json.dumps(result),flush=True)
 except Exception as exc:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
