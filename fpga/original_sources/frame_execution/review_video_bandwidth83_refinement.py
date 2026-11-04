"""Review all2000 paths of the same83 map after congestion-aware refinement."""
import argparse,contextlib,io,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from review_input_isp_geometry import locations
from camera_ratio_before_place import metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 proof=json.loads((root/'refinement-inputs.json').read_text());result=json.loads((root/'refinement-result.json').read_text());base=Path(proof['source'])
 assert result['status']=='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE' and result['exit_code']==0
 assert result['inputs_sha256']==sha(root/'refinement-inputs.json')
 for f,h in proof['evidence'].items():assert sha(f)==h
 for f,h in result['frozen'].items():assert sha(f)==h
 sys.argv=['review_video_bandwidth83.py','--root',str(root)]
 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(Path(__file__).with_name(sys.argv[0])),run_name='__main__')
 pos=[locations(p/'outflow/ti60f225_oob.out.placeloc') for p in (base,root)];assert pos[0].keys()==pos[1].keys()
 report=ROOT/'tmp/frame-execution/video-bandwidth83-canonical1/outflow/canonical_setup.rpt';rows=[]
 for b in re.split(r'(?=^Slack\s*:)',report.read_text(),flags=re.M)[1:]:
  value=lambda pattern:re.search(pattern,b,re.M)[1].strip()
  cells=[];data=re.split(r'^Data Path\s*$',b,flags=re.M)[1].split('Capture Clock Path',1)[0]
  for line in data.splitlines():
   m=re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)',line)
   if m:
    n=m[1].strip().split('|')[0]
    if not cells or n!=cells[-1]:cells.append(n)
  pairs=list(zip(cells,cells[1:]));usable=[(x,y) for x,y in pairs if all(x in q and y in q for q in pos)]
  ds=[sum(abs(q[x][0]-q[y][0])+abs(q[x][1]-q[y][1]) for x,y in usable) for q in pos]
  rows.append(dict(slack_ns=float(value(r'^Slack\s*:\s*([+\-\d.]+)')),start=value(r'^Path Start\s*:\s*(.*)'),end=value(r'^Path End\s*:\s*(.*)'),edges=len(pairs),compared=len(usable),old=ds[0],new=ds[1]))
 assert len(rows)==2000 and all(x['compared']==x['edges'] for x in rows)
 old=json.loads((base/'global-bandwidth-review.json').read_text())['comparison'][1]
 new=json.loads((root/'global-bandwidth-review.json').read_text())['comparison'][1]
 def sums(rs):return dict(count=len(rs),old=sum(v['old'] for v in rs),new=sum(v['new'] for v in rs))
 groups={}
 for r in rows:
  if r['slack_ns']<0:groups.setdefault(r['start'],[]).append(r)
 negative=[r for r in rows if r['slack_ns']<0]
 out=dict(status='BANDWIDTH83_REFINEMENT_REVIEW_NOT_ROUTE_PERMISSION',old_metrics=metrics(base),new_metrics=metrics(root),
  all_paths=sums(rows),negatives=sums(negative),negative_groups={k:sums(v) for k,v in groups.items()},paths=rows,
  old_control_networks=old['control_networks'],new_control_networks=new['control_networks'],
  old_connections=old['connections'],new_connections=new['connections'],
  limitations=['Same-map geometry can compare every path edge but is not actual routed delay.','Original dedicated-clock, protected/alias origin coordinate coverage limits remain; tool STA/CDC required.'],
  evidence={str(f):sha(f) for f in (Path(__file__).resolve(),report,root/'global-bandwidth-review.json',root/'refinement-inputs.json',root/'refinement-result.json',base/'global-bandwidth-review.json')},hardware_access=False,board_release=False)
 (root/'refinement-global-review.json').write_text(json.dumps(out,indent=2))
 display={k:v for k,v in out.items() if k in ('all_paths','negatives','negative_groups')}
 for label in ('old_metrics','new_metrics'):display[label]={k:out[label][k] for k in ('resources','congestion_peaks','weighted_hpwl_per_net','canonical_setup_estimate_ns')}
 print(json.dumps(display,indent=2))
if __name__=='__main__':main()
