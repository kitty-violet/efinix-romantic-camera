"""Whole-chip routability review against the actual released73/82 placement."""
import argparse,collections,json,pickle,re,statistics
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections
from review_input_isp_geometry import locations,group as original_group
from camera_ratio_before_place import metrics
from video_bandwidth83_physical import audit,verify_placed,BOARD

REFERENCE=ROOT/'tmp/frame-execution/video-sensor73-place4-seed1'
OLD_BOARD=ROOT/'tmp/frame-execution/connected-board-video-sensor73-map2'
CLOCK={'CLK','CLKA','CLKB','RCLK','WCLK'}
CONTROL={'SR','CE','RCE','WCE','CEA','CEB','CLKE','RST','WE','WEA','WEB'}
def group(n):
 if n.startswith(('bandwidth83_capture/','bandwidth83_awaddr','camera_burst_capture/','cap_')):return 'capture_writer'
 if n.startswith('u_ddr3_top/address_bridge/'):return 'ddr_bridge'
 if n.startswith('video_processing_pipeline/pipeline/fabric/'):return 'shared_fabric'
 return original_group(n)
def category(p):return 'clock' if p in CLOCK else 'reset_enable' if p in CONTROL else 'data'
def envelope(v):return [min(q[0] for q in v),min(q[1] for q in v),max(q[0] for q in v),max(q[1] for q in v)] if v else None
def distance(pos,a,b):return abs(pos[a][0]-pos[b][0])+abs(pos[a][1]-pos[b][1])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();proof=audit()
 h,g,ps,drivers,loads=pickle.loads((BOARD/'integration-parse.pkl').read_bytes());assert h==sha(root/'outflow/ti60f225_oob.map.v')
 oh,og,ops=pickle.loads((OLD_BOARD/'sensor73-map-parse.pkl').read_bytes());assert oh==sha(REFERENCE/'outflow/ti60f225_oob.map.v')
 source_h,sg,_,_,_=pickle.loads((ROOT/'tmp/board-test/video-gain82-offline1/integration-parse.pkl').read_bytes());assert source_h==sha(ROOT/'tmp/frame-execution/video-sensor73-original-input1/native_source.v')
 prior=json.loads((root/'result.json').read_text());assert prior['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE' and all(s['exit_code']==0 for s in prior['stages'])
 pos=[locations(p/'outflow/ti60f225_oob.out.placeloc') for p in (REFERENCE,root)]
 rows=[]
 for path,loc,graph in zip((REFERENCE,root),pos,(og,g)):
  driver,loads=connections(graph);groups=collections.defaultdict(list);edges=collections.defaultdict(lambda:dict(count=0,distance=0,max_distance=0));controls=[];missing=collections.Counter()
  for n,q in loc.items():groups[group(n)].append(q)
  for token,sinks in loads.items():
   origin=driver.get(token)
   for n,pin in sinks:
    if origin not in loc or n not in loc:missing[category(pin)]+=1;continue
    key=category(pin)+':'+group(origin)+'->'+group(n);d=distance(loc,origin,n);r=edges[key];r['count']+=1;r['distance']+=d;r['max_distance']=max(r['max_distance'],d)
   for kind,ss in [('nonclock',[(n,p) for n,p in sinks if p not in CLOCK]),('clock',[(n,p) for n,p in sinks if p in CLOCK])]:
    if len(ss)<100:continue
    points=[loc[n] for n,p in ss if n in loc]+([loc[origin]] if origin in loc else [])
    controls.append(dict(net=token,driver=origin,kind=kind,fanout=len(ss),pins=dict(collections.Counter(p for n,p in ss)),bbox=envelope(points),missing_cells=sorted({n for n,p in ss if n not in loc})))
  ram={n:list(loc[n]) for n,(k,_) in graph.items() if k in ('EFX_RAM10','EFX_DPRAM10') and n in loc};assert len(ram)==245
  rows.append(dict(physical=str(path),metrics=metrics(path),ram=ram,connections=dict(edges),control_networks=controls,unplaced_edges=dict(missing),groups={n:dict(count=len(v),bbox=envelope(v)) for n,v in groups.items()}))
 # Match old hard-IP names by unique protected prefix; LUT correspondence is
 # accepted only for explicit retained native cells, never inferred LUT IDs.
 bystem=collections.defaultdict(list)
 for n in pos[1]:bystem[re.sub(r'^(?:protected_inst/)+','',n)].append(n)
 matched={}
 for n in pos[0]:
  if n in pos[1] and n in g and (g[n][0]!='EFX_LUT4' or (n in sg and n in og and ps.get(n)==ops.get(n))):matched[n]=n
  elif n not in og:
   choices=bystem.get(re.sub(r'^(?:protected_inst/)+','',n),[])
   if len(choices)==1:matched[n]=choices[0]
 report=ROOT/'tmp/frame-execution/video-sensor73-canonical9/outflow/canonical_setup.rpt';paths=[]
 for block in re.split(r'(?=^Slack\s*:)',report.read_text(),flags=re.M)[1:]:
  slack=float(re.search(r'^Slack\s*:\s*([\d.+-]+)',block)[1]);end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip()
  data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0];cells=[]
  for line in data.splitlines():
   m=re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)',line)
   if m:
    n=m[1].strip().split('|')[0]
    if not cells or n!=cells[-1]:cells.append(n)
  pairs=list(zip(cells,cells[1:]));compared=[(u,v) for u,v in pairs if u in matched and v in matched]
  paths.append(dict(end=end,previous_slack_ns=slack,total_edges=len(pairs),compared_edges=len(compared),previous_geometry=sum(distance(pos[0],u,v) for u,v in compared),candidate_geometry=sum(distance(pos[1],matched[u],matched[v]) for u,v in compared)))
 assert len(paths)==2000,len(paths)
 families=collections.defaultdict(list)
 for r in paths:families[re.sub(r'\[\d+\]','[]',r['end'])].append(r)
 fs=[dict(end=k,count=len(v),min_previous_ns=min(r['previous_slack_ns'] for r in v),previous_geometry=sum(r['previous_geometry'] for r in v),candidate_geometry=sum(r['candidate_geometry'] for r in v)) for k,v in families.items()];fs.sort(key=lambda r:r['min_previous_ns'])
 anchors=verify_placed(root);(root/'hard-anchor-check.json').write_text(json.dumps(anchors,indent=2))
 result=dict(status='BANDWIDTH83_WHOLE_CHIP_PLACEMENT_REVIEW_NOT_ROUTE_PASS',comparison=rows,setup_path_geometry=paths,setup_path_families=fs,
  moved_common_ram=[dict(cell=n,old=xyz,new=rows[1]['ram'][n]) for n,xyz in rows[0]['ram'].items() if n in rows[1]['ram'] and xyz!=rows[1]['ram'][n]],
  replaced_ram_old={n:xyz for n,xyz in rows[0]['ram'].items() if n not in rows[1]['ram']},replaced_ram_new={n:xyz for n,xyz in rows[1]['ram'].items() if n not in rows[0]['ram']},
  limits=['Geometry/congestion estimates are not occupied-track or routed STA proof.','Dedicated clocks and protected internals require fresh original STA/CDC/Gray.','No inferred LUT identity assumed across synthesis. Exact source-scope audit and module tests are separate functional evidence.'],
  evidence={str(f):sha(f) for f in (Path(__file__),BOARD/'bandwidth83-preplace-audit.json',report,root/'physical_inputs.json',root/'outflow/ti60f225_oob.out.placeloc',REFERENCE/'outflow/ti60f225_oob.out.placeloc')},hardware_access=False,board_release=False)
 (root/'global-bandwidth-review.json').write_text(json.dumps(result,indent=2))
 (root/'placement_review_inputs.json').write_text(json.dumps(dict(status='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW',current=rows[1]['metrics'],reference=rows[0]['metrics'],global_review_sha256=sha(root/'global-bandwidth-review.json'),map_sha256=h,hardware_access=False,board_release=False),indent=2))
 print(json.dumps(dict(current=rows[1]['metrics'],reference=rows[0]['metrics'],families=fs[:12]),indent=2))
if __name__=='__main__':main()
