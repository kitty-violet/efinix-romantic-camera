"""Whole-chip and every changed CE branch placement audit before routing."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
from camera_factor_balanced_board import BOARD,PARENT,TRANSFORM,audit_delta
from native_local_drivers import graph
from camera_joint_native import OUT
from fixed_control_ddr_reuse_locations import locations
from camera_ratio_before_place import metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert not a.out.exists();audit_delta(BOARD)
 place=a.placement.resolve();record=json.loads((place/'placement_review_inputs.json').read_text());assert record['status']=='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW'
 oldplace=ROOT/'tmp/frame-execution/camera-prefetch-place1';selected=ROOT/'tmp/frame-execution/camera-recovery-place1'
 current=metrics(place);before=metrics(oldplace);baseline=metrics(selected)
 og=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());ng=graph((BOARD/'native_source.v').read_text());oldxy=locations(oldplace/'outflow/ti60f225_oob.out.placeloc');xy=locations(place/'outflow/ti60f225_oob.out.placeloc')
 def drivers(g):return {t:n for n,(k,p) in g.items() for pin,ts in p.items() if pin in OUT.get(k,[]) for t in ts}
 od,nd=drivers(og),drivers(ng)
 proof=json.loads((TRANSFORM/'result.json').read_text());changed={n for s in proof['factor_stages'] for r in s['outputs'] for n,pin in r['loads']}
 grouped=collections.defaultdict(list)
 for n in changed:grouped[(og[n][1]['CE'][0],ng[n][1]['CE'][0])].append(n)
 # No old physically distinct CE outputs may merge at the final sink.
 seen={}
 for old,new in grouped:
  assert new not in seen or seen[new]==old,(old,new,seen.get(new))
  seen[new]=old
 def dist(pos,s,d):return abs(pos[s][0]-pos[d][0])+abs(pos[s][1]-pos[d][1])
 rows=[]
 for (old,new),sinks in sorted(grouped.items(),key=lambda z:str(z[0])):
  x,y=od[old],nd[new];valid=[n for n in sinks if n in xy and n in oldxy]
  assert len(valid)==len(sinks),(x,len(valid),len(sinks))
  b=[dist(oldxy,x,n) for n in valid];c=[dist(xy,y,n) for n in valid]
  rows.append(dict(old_net=old,new_net=new,before_driver=x,after_driver=y,loads=len(valid),before_mean=sum(b)/len(b),after_mean=sum(c)/len(c),before_max=max(b),after_max=max(c)))
 controls=[]
 for s in proof['factor_stages']:
  root=tuple(s['root']);d=nd[root];loads=[]
  for n,(k,pins) in ng.items():
   if n in xy and any(root in ts for pin,ts in pins.items() if pin not in OUT.get(k,[])):loads.append(n)
  ds=[dist(xy,d,n) for n in loads]
  controls.append(dict(root=root,driver=d,loads=len(loads),mean=sum(ds)/len(ds),maximum=max(ds)))
 def diff(ref):return dict(hpwl_pct=100*(current['weighted_hpwl_per_net']/ref['weighted_hpwl_per_net']-1),**{k:100*(current['congestion_peaks'][k]/v-1) for k,v in ref['congestion_peaks'].items()})
 res=current['resources'];fit=all(res[k]['used']<=limit for k,limit in [('XLRs',60800),('Memory Blocks',256),('DSP Blocks',160)])
 files=[BOARD/'result.json',TRANSFORM/'result.json',place/'physical_inputs.json',place/'result.json',place/'placement_review_inputs.json',place/'outflow/ti60f225_oob.out.placeloc',place/'fixed_dsp.placeloc',Path(__file__)]
 files += [ROOT/'tmp/board-test'/n for n in ('camera-factor-balanced-loads1.json','camera-factor-balanced-release-preflight1.json')]
 result=dict(status='FACTOR_WHOLE_CHIP_PLACEMENT_REVIEW_NOT_ROUTE_APPROVAL',current=current,previous_prefetch=before,selected_parent=baseline,relative_to_prefetch=diff(before),relative_to_selected=diff(baseline),capacity_fit=fit,original_ce_groups_preserved=True,changed_ce_loads=len(changed),ce_branches=rows,late_controls=controls,global_graph=record['graph'],geometry_scope='Complete changed-CE branch distances and source loads; geometry is not routed delay or a guarantee of routability. No added clocks/resets or individually fixed ordinary logic.',evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 a.out.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:result[k] for k in ('status','relative_to_prefetch','relative_to_selected','capacity_fit','original_ce_groups_preserved','late_controls')}))
 print(json.dumps({k:current[k] for k in ('resources','congestion_peaks','weighted_hpwl_per_net','canonical_setup_estimate_ns')}))
 print('CE_BRANCH_MAXIMA',json.dumps(sorted(rows,key=lambda r:r['after_max'],reverse=True)[:5]))
if __name__=='__main__':main()
