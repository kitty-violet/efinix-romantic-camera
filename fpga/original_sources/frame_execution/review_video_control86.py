"""Global placement review plus actual failed83 paths with the84 shortcut."""
import collections,contextlib,io,json,pickle,re,sys
from pathlib import Path
import review_video_control86_global as base
from video_control86_physical import audit,verify_placed,BOARD
from run_units import ROOT,sha
from review_input_isp_geometry import locations
from camera_ratio_before_place import metrics
from continuous_control_native import connections

def main():
 base.audit=audit;base.verify_placed=verify_placed;base.BOARD=BOARD
 with contextlib.redirect_stdout(io.StringIO()):base.main()
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 previous=ROOT/'tmp/frame-execution/video-control84-place1'
 canonical=ROOT/'tmp/frame-execution/video-control84-canonical1/outflow/canonical_setup.rpt'
 h,g,ps,_,_=pickle.loads((BOARD/'integration-parse.pkl').read_bytes());assert h==sha(root/'outflow/ti60f225_oob.map.v')
 old_board=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'
 oh,og,ops,_,_=pickle.loads((old_board/'integration-parse.pkl').read_bytes());assert oh==sha(previous/'outflow/ti60f225_oob.map.v')
 pos0=locations(previous/'outflow/ti60f225_oob.out.placeloc');pos1=locations(root/'outflow/ti60f225_oob.out.placeloc')
 dr,ld=connections(g);arcs={(dr[t],n) for t,sinks in ld.items() if t in dr for n,p in sinks}
 rows=[]
 for b in re.split(r'(?=^Slack\s*:)',canonical.read_text(),flags=re.M)[1:]:
  slack=float(re.search(r'^Slack\s*:\s*([\d.+-]+)',b,re.M)[1]);end=re.search(r'^Path End\s*:\s*(.*)',b,re.M)[1].strip()
  data=re.split(r'^Data Path\s*$',b,flags=re.M)[1].split('Capture Clock Path',1)[0];cells=[]
  for line in data.splitlines():
   m=re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)',line)
   if m:
    n=m[1].strip().split('|')[0]
    if not cells or n!=cells[-1]:cells.append(n)
  current=list(cells);shortcut=False
  if all(n in current for n in ('ENDPOINT_BANK_ADDRESS_0','CONTROL_REMAP_1152')):
   a=current.index('ENDPOINT_BANK_ADDRESS_0');z=current.index('CONTROL_REMAP_1152')
   segment=current[a+1:z];assert segment in (['SERVICE64_STOP_23','SERVICE64_STOP_24','SERVICE64_STOP_25'],['SERVICE64_STOP_27','SERVICE64_STOP_28','SERVICE64_STOP_29']),segment
   current=current[:a+1]+[segment[0],'BANDWIDTH_CONTROL86_HR']+current[z:];shortcut=True
  original_edges=list(zip(cells,cells[1:]));current_edges=list(zip(current,current[1:]))
  missing=[(u,v) for u,v in current_edges if u not in pos1 or v not in pos1]
  # Synthesis can reuse auto-generated LUT names for a different function.
  # Geometry from those cells is not a defensible cross-map correspondence.
  uncertain=[n for n in current if n not in ('CONTROL_REMAP_1152',) and n in g and n in og
   and g[n][0]=='EFX_LUT4' and (g[n]!=og[n] or ps.get(n)!=ops.get(n))]
  # Protected and special arithmetic connections are not all represented by
  # this public graph. Explicitly qualify every graph-covered new shortcut.
  if shortcut:assert all(edge in arcs for edge in [('ENDPOINT_BANK_ADDRESS_0',segment[0]),(segment[0],'BANDWIDTH_CONTROL86_HR'),('BANDWIDTH_CONTROL86_HR','CONTROL_REMAP_1152')])
  def length(edges,pos):return sum(abs(pos[u][0]-pos[v][0])+abs(pos[u][1]-pos[v][1]) for u,v in edges if u in pos and v in pos)
  rows.append(dict(end=end,prior_slack_ns=slack,shortcut=shortcut,matched=not missing and not uncertain,old_cells=len(cells),candidate_cells=len(current),missing=missing,uncertain_lut_identity=uncertain,
   previous_geometry=length(original_edges,pos0),candidate_geometry=length(current_edges,pos1)))
 assert len(rows)==2000
 neg=[r for r in rows if r['prior_slack_ns']<0]
 assert len(neg)==2 and sum(r['shortcut'] for r in neg)==2 and all(r['matched'] for r in neg)
 matched=[r for r in rows if r['matched']]
 result=dict(status='CONTROL86_WHOLE_CHIP_AND_REAL_PATH_REVIEW_NOT_ROUTE_PASS',
  previous_metrics=metrics(previous),candidate_metrics=metrics(root),negative_paths=neg,all_paths=rows,
  old_negative_geometry=sum(r['previous_geometry'] for r in neg),candidate_negative_geometry=sum(r['candidate_geometry'] for r in neg),
  all_paths_count=len(rows),matched_paths_count=len(matched),old_all_geometry=sum(r['previous_geometry'] for r in matched),candidate_all_geometry=sum(r['candidate_geometry'] for r in matched),
  limits=['Logic-shortened paths are actual public graph arcs, but no propagation delay is inferred from geometry.','Other input cones to factored logic and newly critical paths require fresh all-path STA.','Global congestion/control metrics must be considered before route.'],
  evidence={str(f):sha(f) for f in (Path(__file__),Path(base.__file__),BOARD/'control86-preplace-audit.json',old_board/'outflow/ti60f225_oob.map.v',canonical,root/'global-bandwidth-review.json',root/'outflow/ti60f225_oob.out.placeloc')},hardware_access=False,board_release=False)
 (root/'control86-path-review.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:result[k] for k in ('status','previous_metrics','candidate_metrics','old_negative_geometry','candidate_negative_geometry','old_all_geometry','candidate_all_geometry')},indent=2))
if __name__=='__main__':main()
