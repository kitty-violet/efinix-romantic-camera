"""Whole-chip placement and per-group geometry review; never starts routing."""
import argparse,json,collections
from pathlib import Path
from run_units import ROOT,sha
from camera_joint_board import BOARD,PARENT,TRANSFORM
from camera_joint_native import OUT
from native_local_drivers import graph
from fixed_control_ddr_reuse_locations import locations
from camera_ratio_before_place import metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();place=a.placement.resolve();assert not a.out.exists()
 r=json.loads((place/'placement_review_inputs.json').read_text());assert r['status']=='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW'
 oldplace=ROOT/'tmp/frame-execution/camera-prefetch-place1';selected=ROOT/'tmp/frame-execution/camera-recovery-place1'
 before=metrics(oldplace);current=metrics(place);baseline=metrics(selected)
 def diff(ref):return dict(hpwl_pct=100*(current['weighted_hpwl_per_net']/ref['weighted_hpwl_per_net']-1),**{k:100*(current['congestion_peaks'][k]/v-1) for k,v in ref['congestion_peaks'].items()})
 proof=json.loads((TRANSFORM/'result.json').read_text());src=graph((TRANSFORM/'native_source.v').read_text());oldxy=locations(oldplace/'outflow/ti60f225_oob.out.placeloc');newxy=locations(place/'outflow/ti60f225_oob.out.placeloc')
 drivers={t:n for n,(k,p) in src.items() for pin,ts in p.items() if pin in OUT.get(k,[]) for t in ts}
 def distance(xy,s,d):return abs(xy[s][0]-xy[d][0])+abs(xy[s][1]-xy[d][1])
 rows=[]
 for rec in proof['replicas']:
  name=rec['driver'];b=[];c=[];ordinary=[]
  for target,pin in rec['original_loads']:
   if target in oldxy and target in newxy:b.append(distance(oldxy,name,target));c.append(distance(newxy,name,target))
  for clone in rec['clones']:
   driver=drivers[tuple(clone['output'])];dist=[]
   for target,pin in clone['loads']:
    b.append(distance(oldxy,name,target));d=distance(newxy,driver,target);c.append(d);dist.append(d)
   ordinary.extend(dist)
  rows.append(dict(driver=name,net=rec['net'],before_mean=sum(b)/len(b),after_mean=sum(c)/len(c),before_max=max(b),after_max=max(c),cloned_branch_max=max(ordinary),cloned_branch_mean=sum(ordinary)/len(ordinary),edges=len(b)))
 res=current['resources'];fit=all(res[n]['used']<=lim for n,lim in [('XLRs',60800),('Memory Blocks',256),('DSP Blocks',160)])
 evidence=[BOARD/'result.json',TRANSFORM/'result.json',ROOT/'tmp/board-test/camera-joint-loads2.json',ROOT/'tmp/board-test/camera-joint-release-preflight1.json',ROOT/'tmp/frame-execution/camera-joint-evidence2/result.json']
 evidence += [place/n for n in ('physical_inputs.json','result.json','placement_review_inputs.json','cnn_ram_actual_locations.json','cnn_ram_placement.json','fixed_dsp.placeloc','outflow/ti60f225_oob.out.placeloc')]
 result=dict(status='JOINT_WHOLE_CHIP_PLACEMENT_REVIEW_NOT_ROUTE_APPROVAL',current=current,previous_prefetch=before,selected_parent=baseline,relative_to_prefetch=diff(before),relative_to_selected=diff(baseline),capacity_fit=fit,
  affected_geometry=rows,global_graph=r['graph'],geometry_scope='Manhattan distance estimates for original consumer edges only; not routed delay, track capacity or attribution of actual conflicts. New upstream load increases are independently inventoried.',
  evidence={str(f):sha(f) for f in evidence+[Path(__file__)]},hardware_access=False,board_release=False)
 a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','relative_to_prefetch','relative_to_selected','capacity_fit')}));print(json.dumps({k:current[k] for k in ('resources','congestion_peaks','weighted_hpwl_per_net','canonical_setup_estimate_ns')}))
 print(json.dumps(sorted(rows,key=lambda r:r['before_mean']-r['after_mean'],reverse=True)[:6]));print('WORSENED',json.dumps([r for r in rows if r['after_mean']>r['before_mean']]))
if __name__=='__main__':main()
