"""Whole-chip geometry/control comparison before routing the scoped sensor edit."""
import argparse,collections,json,pickle,re,statistics
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections
from review_input_isp_geometry import locations,group
from camera_ratio_before_place import metrics
from video_sensor73_physical import verify_placed,BOARD,REFERENCE

CLOCK={'CLK','CLKA','CLKB','RCLK','WCLK'}
CONTROL={'SR','CE','RCE','WCE','CEA','CEB','CLKE','RST','WE','WEA','WEB'}

def category(pin):return 'clock' if pin in CLOCK else 'reset_enable' if pin in CONTROL else 'data'
def envelope(v):
 return [min(q[0] for q in v),min(q[1] for q in v),max(q[0] for q in v),max(q[1] for q in v)] if v else None
def distance(pos,a,b):return abs(pos[a][0]-pos[b][0])+abs(pos[a][1]-pos[b][1])

def run(root):
 h,g,params=pickle.loads((BOARD/'sensor73-map-parse.pkl').read_bytes())
 assert h==sha(BOARD/'outflow/ti60f225_oob.map.v')==sha(root/'outflow/ti60f225_oob.map.v')
 proof=json.loads((BOARD/'sensor73-map-audit.json').read_text())
 assert proof['status']=='SENSOR73_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 assert sha(REFERENCE/'outflow/ti60f225_oob.map.v')=='21bfdc8a33626feed493cddc74fd73f7c191c7ad33346c2c520f6a6582407c26'
 assert json.loads((root/'result.json').read_text())['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
 key,source_graph,old_graph,_,_=pickle.loads((BOARD.parent/'connected-board-video-service-probe60-input-isp-explicit/retained-audit-parse.pkl').read_bytes())
 assert key[1]==sha(REFERENCE/'outflow/ti60f225_oob.map.v')
 positions=[locations(p/'outflow/ti60f225_oob.out.placeloc') for p in (REFERENCE,root)]
 target_by_stem=collections.defaultdict(list)
 for n in positions[1]:target_by_stem[re.sub(r'^(?:protected_inst/)+','',n)].append(n)
 matched={}
 for n in positions[0]:
  if n in g and n in positions[1] and (g[n][0]!='EFX_LUT4' or n in source_graph):matched[n]=n
  elif n not in old_graph:
   candidates=target_by_stem.get(re.sub(r'^(?:protected_inst/)+','',n),[])
   if len(candidates)==1:matched[n]=candidates[0]
 rows=[]
 for path,pos,current_graph in zip((REFERENCE,root),positions,(old_graph,g)):
  driver,loads=connections(current_graph)
  groups=collections.defaultdict(list)
  for n,q in pos.items():groups[group(n)].append(q)
  edges=collections.defaultdict(lambda:dict(count=0,distance=0,max_distance=0))
  controls=[];missing=collections.Counter()
  for token,sinks in loads.items():
   origin=driver.get(token)
   for target,pin in sinks:
    if origin not in pos or target not in pos:
     missing[category(pin)]+=1;continue
    key=category(pin)+':'+group(origin)+'->'+group(target);d=distance(pos,origin,target)
    item=edges[key];item['count']+=1;item['distance']+=d;item['max_distance']=max(item['max_distance'],d)
   nonclock=[(n,p) for n,p in sinks if p not in CLOCK]
   clocks=[(n,p) for n,p in sinks if p in CLOCK]
   for kind,ss in [('nonclock',nonclock),('clock',clocks)]:
    if len(ss)<100:continue
    points=[pos[n] for n,p in ss if n in pos]+([pos[origin]] if origin in pos else [])
    controls.append(dict(net=token,driver=origin,kind=kind,fanout=len(ss),
     pins=dict(collections.Counter(p for n,p in ss)),bbox=envelope(points),
     missing_cells=sorted({n for n,p in ss if n not in pos})))
  ram={n:list(pos[n]) for n,(k,_) in current_graph.items() if k in ('EFX_RAM10','EFX_DPRAM10') and n in pos}
  assert len(ram)==245
  rows.append(dict(physical=str(path),metrics=metrics(path),ram=ram,connections=dict(edges),
   groups={n:dict(count=len(v),bbox=envelope(v),median_xy=[statistics.median(q[i] for q in v) for i in range(2)]) for n,v in groups.items()},
   control_networks=controls,unplaced_edges=dict(missing)))
 changes=[dict(cell=n,group=group(n),old=loc,new=rows[1]['ram'][n]) for n,loc in rows[0]['ram'].items() if loc!=rows[1]['ram'][n]]
 # Revisit every reported path in the passing implementation, including small positive margins.
 report=ROOT/'tmp/frame-execution/video-input-canonical2/outflow/canonical_setup.rpt'
 paths=[]
 for block in re.split(r'(?=^Slack\s*:)',report.read_text(),flags=re.M)[1:]:
  slack=float(re.search(r'^Slack\s*:\s*([\d.+-]+)',block)[1]);end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip()
  data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0];cells=[]
  for line in data.splitlines():
   m=re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)',line)
   if m:
    n=m[1].strip().split('|')[0]
    if not cells or n!=cells[-1]:cells.append(n)
  pairs=list(zip(cells,cells[1:]));compared=[(u,v) for u,v in pairs if u in matched and v in matched]
  paths.append(dict(end=end,previous_slack_ns=slack,total_edges=len(pairs),compared_edges=len(compared),
   previous_geometry=sum(distance(positions[0],u,v) for u,v in compared),candidate_geometry=sum(distance(positions[1],matched[u],matched[v]) for u,v in compared)))
 assert len(paths)==2000,len(paths)
 families=collections.defaultdict(list)
 for x in paths:families[re.sub(r'\[\d+\]','[]',x['end'])].append(x)
 family_rows=[dict(end=k,count=len(v),min_previous_ns=min(x['previous_slack_ns'] for x in v),
  previous_geometry=sum(x['previous_geometry'] for x in v),candidate_geometry=sum(x['candidate_geometry'] for x in v)) for k,v in families.items()]
 family_rows.sort(key=lambda r:r['min_previous_ns'])
 anchor=verify_placed(root)
 (root/'hard-anchor-check.json').write_text(json.dumps(anchor,indent=2),encoding='utf8')
 result=dict(status='SENSOR73_WHOLE_CHIP_PLACEMENT_REVIEW_NOT_ROUTE_PASS',comparison=rows,moved_ram=changes,
  setup_path_geometry=paths,setup_path_families=family_rows,protected_path_name_rebindings={k:v for k,v in matched.items() if k!=v},new_cells={n:list(positions[1][n]) for n in g if n.startswith('sensor_rom_expected_mux')},
  limits=['Geometry is not routed timing or occupied track evidence.','Clock, reset/enable and ordinary data are reported separately; actual new-route CDC/Gray remains required.',
   'Each placement uses its own mapped graph. Path correspondence excludes inferred LUT names without proven stable source identity; compared-edge counts expose exclusions. Eight source muxes and changed sensor edges have no old placement equivalence.',
   'Public mapped primitives omit protected hard-IP internals; exact protected source and pins remain audited separately.'],
  evidence={str(f):sha(f) for f in (Path(__file__),BOARD/'sensor73-map-audit.json',report,root/'physical_inputs.json',
   root/'hard-anchor-check.json',root/'outflow/ti60f225_oob.out.placeloc',REFERENCE/'outflow/ti60f225_oob.out.placeloc')},hardware_access=False,board_release=False)
 (root/'global-sensor-review.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 (root/'placement_review_inputs.json').write_text(json.dumps(dict(status='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW',current=rows[1]['metrics'],reference=rows[0]['metrics'],
  global_review_sha256=sha(root/'global-sensor-review.json'),map_sha256=h,hardware_access=False,board_release=False),indent=2),encoding='utf8')
 print(json.dumps(dict(status=result['status'],current=rows[1]['metrics'],reference=rows[0]['metrics'],moved_ram=len(changes),families=family_rows[:12])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
