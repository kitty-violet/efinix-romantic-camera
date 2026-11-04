"""One official TIMING_1 comparison on the exact legal split-output placement."""
import argparse,json,re,runpy,shutil,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from split_logic_outputs_board import audit_delta

BASE=ROOT/'tmp/frame-execution'
OLD=BASE/'split-outputs-place3'
PHYSICAL=BASE/'split-outputs-timing-place1'
BATCH=BASE/'split-outputs-timing-route1'
BOARD=BASE/'connected-board-split-outputs1'

def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')

def verify(rows):
 for p,h in rows.items():assert sha(p)==h,p

def native_paths(file):
 rows=[]
 for block in re.split(r'(?=^Path Begin\s*:)',file.read_text(),flags=re.M)[1:]:
  slack=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',block,re.M)[1])
  if slack>=0:continue
  data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]
  nets=[dict(net=n,delay_ns=float(t)) for n,t in re.findall(r'^\s*(\S+)\s+net\s+([\d.]+)\s+[\d.]+\s+\d+\s+\(',data,re.M)]
  assert nets
  rows.append(dict(slack_ns=slack,start=re.search(r'^Path Begin\s*:\s*(.*)',block,re.M)[1].strip(),
   end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip(),
   data_net_total_ns=sum(n['delay_ns'] for n in nets),longest_net=max(nets,key=lambda n:n['delay_ns'])))
 assert len(rows)==10 and rows[0]['slack_ns']==-.735
 assert rows[0]['longest_net']['delay_ns']==3.721
 return dict(status='NATIVE_TOP_TEN_NEGATIVE_PATHS_ANALYZED',report=str(file),report_sha256=sha(file),
  rows=rows,limitation='Ten reported worst paths; not the count of all failing endpoints.',hardware_access=False)

def prepare():
 assert not PHYSICAL.exists() and not BATCH.exists()
 audit_delta(BOARD)
 selected=json.loads((BASE/'split-outputs-route1/selected.json').read_text())
 assert selected['wns_ns']==-.735 and selected['whs_ns']==.026
 origin=json.loads((OLD/'route_review.json').read_text())
 evidence=dict(origin['required_evidence'])
 # The reviewed placement result is archived verbatim by the first router.
 expected=evidence.pop(str(OLD/'result.json'))
 assert sha(OLD/'before_route/result.json')==expected
 evidence[str(OLD/'before_route/result.json')]=expected
 verify(evidence)
 meta=json.loads((OLD/'physical_inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 exclusions={'before_route','route_only.cmd','route_review.json','result.json','stages.json',
  'ti60f225_oob.troutingtraces','ti60f225_oob.route.out','ti60f225_oob.route.rpt',
  'ti60f225_oob.route.rpt.xml','ti60f225_oob.timing.rpt','ti60f225_oob.hold_fixing.rpt'}
 shutil.copytree(OLD,PHYSICAL,ignore=lambda directory,names:[n for n in names if n in exclusions])
 names=['outflow/ti60f225_oob'+s for s in ('.map.v','.place','.gplace','.out.placeloc','.vdb','.interface.csv')]
 names+=['ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','efx_pnr_settings.ini','fixed_dsp.placeloc',
  'work_pnr/ti60f225_oob.net_proto','work_pnr/ti60f225_oob.primplace','work_pnr/ti60f225_oob.io_place']
 original={str(OLD/n):sha(OLD/n) for n in names};copied={str(PHYSICAL/n):sha(PHYSICAL/n) for n in names}
 assert list(original.values())==list(copied.values())
 for row in meta['inputs']:
  row['snapshot']=str(PHYSICAL/Path(row['snapshot']).relative_to(OLD))
  assert sha(row['snapshot'])==row['sha256']
 save(PHYSICAL/'physical_inputs.json',meta)
 shutil.copy2(OLD/'before_route/result.json',PHYSICAL/'result.json')
 save(PHYSICAL/'stages.json',json.loads((PHYSICAL/'result.json').read_text())['stages'])
 for suffix in ('-place-driver.py','-place-driver.json','-eight-thread-wrapper.py','-eight-thread-wrapper.json'):
  shutil.copy2(BASE/(OLD.name+suffix),BASE/(PHYSICAL.name+suffix))
 inheritance=dict(status='EXACT_PLACEMENT_REUSED_FOR_ROUTE_PROFILE_COMPARISON',source=str(OLD),
  source_physical_inputs_sha256=sha(OLD/'physical_inputs.json'),source_files=original,copied_files=copied,
  source_placement_result_sha256=sha(OLD/'before_route/result.json'),
  source_route_result=str(BASE/'split-outputs-route1/selected.json'),
  source_route_result_sha256=sha(BASE/'split-outputs-route1/selected.json'),
  generator=str(Path(__file__)),generator_sha256=sha(__file__),placement_executed=False,
  rtl_changed=False,constraints_changed=False,seed_changed=False,
  route_profile_cli_override=dict(before='CONGESTION_2',after='TIMING_1'),hardware_access=False)
 save(PHYSICAL/'route_inheritance.json',inheritance)
 save(PHYSICAL/'data_arc_analysis.json',native_paths(OLD/'outflow/ti60f225_oob.timing.rpt'))
 for f in [PHYSICAL/'physical_inputs.json',PHYSICAL/'route_inheritance.json',PHYSICAL/'data_arc_analysis.json',
  Path(__file__),HERE/'split_logic_outputs_route.py',EF/'bin/efx_pnr_options.xml',
  EF/'scripts/aoe/core/options_constants.py',EF/'doc/pdf/efinity-timing-closure.pdf']:
  evidence[str(f)]=sha(f)
 save(PHYSICAL/'route_review.json',dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(PHYSICAL),
  placement_review_sha256=sha(PHYSICAL/'placement_review_inputs.json'),
  rationale='Reuse all cells of the legal 56128-XLR C2 placement and the same fully checked zero-COMB4 map. Seven of ten worst reported paths converge on bd[536] with 3.721ns net delay; the other three also contain 3.336-4.064ns data nets. Test one official timing-oriented route profile without changing functional latency, clock, constraints or placement. The first route is legal at 71 iterations. This is one controlled profile comparison, not seed sweeping.',
  remaining_risks=['92.32% logic density and worse placement congestion remain unchanged.',
   'An older different 56501-XLR map failed TIMING_1 routing; official documentation does not rank recipes or guarantee improvement.',
   'Current original STA fails. This comparison must obtain new legal routing, original setup/hold, CDC and release evidence.'],
  required_evidence=evidence,budget_seconds=1250,host_budget_seconds=1800,route_attempts_authorized=1,
  source_and_constraints_unchanged=True,hardware_access=False,board_release=False))
 print(json.dumps(dict(status='EXACT_PLACEMENT_TIMING_COMPARISON_PREPARED',physical=str(PHYSICAL),batch=str(BATCH))),flush=True)

def execute():
 # Transform the already audited split-output route wrapper in a new file.
 # Every existing mapped-short/physical/original-STA/Gray gate remains present.
 source=HERE/'split_logic_outputs_route.py';code=source.read_text()
 anchor=' for before,after in changes.items():'
 inject=''' changes['--pnr_opts route max_router_iterations=100 troute_options=max_iterations=100']='--pnr_opts route max_router_iterations=100 troute_options=max_iterations=100 optimization_level=TIMING_1'
 changes["        ps=old['stages']+"]="        assert 'optimization_level=TIMING_1' in (root/'route.log').read_text()\\n        inheritance=json.loads((physical/'route_inheritance.json').read_text())\\n        for section in ('source_files','copied_files'):\\n            for path,h in inheritance[section].items():assert sha(path)==h,path\\n        ps=old['stages']+"
'''
 assert code.count(anchor)==1;code=code.replace(anchor,inject+anchor)
 driver=BASE/(BATCH.name+'-wrapper.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 save(driver.with_suffix('.json'),dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),
  generator=str(Path(__file__)),generator_sha256=sha(__file__)))
 sys.argv=[str(driver),'--physical',str(PHYSICAL),'--root',str(BATCH),'--review',str(PHYSICAL/'route_review.json'),
  '--evidence-root',str(BASE/'split-outputs-evidence1'),'--serial-proof',str(BASE/'split-outputs-evidence1-serial2/new'),'--budget','1250']
 runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
 prepare() if a.action=='prepare' else execute()
