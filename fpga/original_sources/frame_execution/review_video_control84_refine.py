"""Equal-map whole-chip and 2000 routed-path review before new route."""
import argparse,collections,contextlib,io,json,pickle,re,sys
from pathlib import Path
from run_units import ROOT,sha
from video_control84_physical import BOARD,audit,verify_placed
from review_input_isp_geometry import locations
from review_video_control84_global import group,category,envelope
from camera_ratio_before_place import metrics
import review_video_control84_global as global_review

BASE=ROOT/'tmp/frame-execution/video-control84-place1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 audit();assert json.loads((root/'refinement-result.json').read_text())['status']=='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE'
 with contextlib.redirect_stdout(io.StringIO()):global_review.main()
 h,g,ps,drivers,loads=pickle.loads((BOARD/'integration-parse.pkl').read_bytes())
 assert h==sha(BASE/'outflow/ti60f225_oob.map.v')==sha(root/'outflow/ti60f225_oob.map.v')
 old,new=[locations(d/'outflow/ti60f225_oob.out.placeloc') for d in (BASE,root)]
 assert set(old)==set(new),'Same mapped implementation must retain every placed cell'
 def dist(pos,u,v):return abs(pos[u][0]-pos[v][0])+abs(pos[u][1]-pos[v][1])
 edge_summary=collections.defaultdict(lambda:dict(edges=0,old=0,new=0,worse=0,better=0))
 net_summary=[];unplaced=collections.Counter()
 for token,sinks in loads.items():
  driver=drivers.get(token);prior=after=matched=0
  for n,pin in sinks:
   k=category(pin)+':'+group(driver or '')+'->'+group(n)
   if driver not in old or n not in old:unplaced[category(pin)]+=1;continue
   x=dist(old,driver,n);y=dist(new,driver,n);r=edge_summary[k]
   r['edges']+=1;r['old']+=x;r['new']+=y;r['worse']+=y>x;r['better']+=y<x
   prior+=x;after+=y;matched+=1
  if len(sinks)>=100:
   net_summary.append(dict(net=token,driver=driver,fanout=len(sinks),pins=dict(collections.Counter(p for n,p in sinks)),
    matched_edges=matched,old_distance=prior,new_distance=after,
    old_bbox=envelope([old[n] for n,p in sinks if n in old]),new_bbox=envelope([new[n] for n,p in sinks if n in new])))
 all_paths=[];refs=[]
 for label in ('video-control84-canonical1','video-control84-canonical2-margin'):
  report=ROOT/'tmp/frame-execution'/label/'outflow/canonical_setup.rpt';refs.append(report);rows=[]
  for b in re.split(r'(?=^Slack\s*:)',report.read_text(),flags=re.M)[1:]:
   slack=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',b)[1]);data=re.split(r'^Data Path\s*$',b,flags=re.M)[1].split('Capture Clock Path',1)[0];cells=[]
   for line in data.splitlines():
    m=re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)',line)
    if m:
     n=m[1].strip().split('|')[0]
     if not cells or n!=cells[-1]:cells.append(n)
   assert cells and all(n in old and n in new for n in cells)
   edges=list(zip(cells,cells[1:]));rows.append(dict(start=re.search(r'^Path Start\s*:\s*(.*)',b,re.M)[1].strip(),
    end=re.search(r'^Path End\s*:\s*(.*)',b,re.M)[1].strip(),prior_slack_ns=slack,
    old_geometry=sum(dist(old,u,v) for u,v in edges),new_geometry=sum(dist(new,u,v) for u,v in edges)))
  assert len(rows)==2000
  negative=[r for r in rows if r['prior_slack_ns']<0]
  families=collections.defaultdict(list)
  for r in rows:families[re.sub(r'\[\d+\]','[]',r['end'])].append(r)
  fam=[dict(end=k,count=len(v),min_previous_ns=min(r['prior_slack_ns'] for r in v),
   old=sum(r['old_geometry'] for r in v),new=sum(r['new_geometry'] for r in v)) for k,v in families.items()]
  all_paths.append(dict(reference=label,rows=rows,negative=negative,negative_old=sum(r['old_geometry'] for r in negative),
   negative_new=sum(r['new_geometry'] for r in negative),all_old=sum(r['old_geometry'] for r in rows),all_new=sum(r['new_geometry'] for r in rows),
   families=sorted(fam,key=lambda r:(r['min_previous_ns'],r['end']))))
 result=dict(status='CONTROL84_C2_MARGIN_EQUAL_MAP_GLOBAL_REVIEW_NOT_ROUTE_PASS',map_sha256=h,
  prior_metrics=metrics(BASE),new_metrics=metrics(root),equal_graph_edges=dict(edge_summary),unplaced_edges=dict(unplaced),
  high_fanout_networks=net_summary,path_sets=all_paths,hard_anchors=verify_placed(root),
  moved_cells=sum(old[n]!=new[n] for n in old),placed_cells=len(old),ordinary_anchors=0,
  moved_ram=[dict(cell=n,old=old[n],new=new[n]) for n,(kind,_) in g.items() if kind in ('EFX_RAM10','EFX_DPRAM10') and n in old and old[n]!=new[n]],
  evidence={str(f):sha(f) for f in [Path(__file__),BOARD/'control84-preplace-audit.json',
   root/'refinement-inputs.json',root/'refinement-result.json',root/'global-bandwidth-review.json',
   BASE/'outflow/ti60f225_oob.out.placeloc',root/'outflow/ti60f225_oob.out.placeloc',*refs]},
  limits=['Geometry is not routed delay. Every placed cell and map graph is identical between compared implementations, so equal edge coverage is exact.',
   'Two2000-path samples are independently compared; they overlap and are not4000 unique endpoints.',
   'Setup placement estimate includes+.300 optimization pressure. Independent final acceptance restores original SDC.'],hardware_access=False,board_release=False)
 (root/'control84-path-review.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status=result['status'],prior_metrics=result['prior_metrics'],new_metrics=result['new_metrics'],
  moved_cells=result['moved_cells'],paths=[{k:r[k] for k in ('reference','negative','negative_old','negative_new','all_old','all_new')} for r in all_paths]),indent=2))
if __name__=='__main__':main()
