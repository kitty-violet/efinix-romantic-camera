"""Check placement cost transferred to replica inputs; no route prediction."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
from camera_targeted_board import BOARD,PARENT,TRANSFORM
from camera_joint_native import OUT
from native_local_drivers import graph
from fixed_control_ddr_reuse_locations import locations

def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();place=a.placement.resolve();assert not a.out.exists()
 proof=json.loads((TRANSFORM/'result.json').read_text())
 before=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());after=graph((BOARD/'outflow/ti60f225_oob.map.v').read_text())
 prior=ROOT/'tmp/frame-execution/camera-prefetch-place1'
 oldxy=locations(prior/'outflow/ti60f225_oob.out.placeloc');newxy=locations(place/'outflow/ti60f225_oob.out.placeloc')
 def drivers(g):return {t:n for n,(k,pins) in g.items() for pin,ts in pins.items() if pin in OUT.get(k,[]) for t in ts}
 olddrivers,newdrivers=drivers(before),drivers(after)
 def dist(xy,s,d):return abs(xy[s][0]-xy[d][0])+abs(xy[s][1]-xy[d][1])
 rows=[];external=[];clock_loads=0
 for clone,original in proof['clone_orig'].items():
  k,np=after[clone];ok,op=before[original];assert k==ok
  for pin,ts in np.items():
   if pin in OUT[k]:continue
   assert len(ts)==len(op[pin]),(clone,pin)
   for t,ot in zip(ts,op[pin]):
    if t[0]=='literal':continue
    if pin=='CLK':clock_loads+=1;continue
    ns=newdrivers.get(t);os=olddrivers.get(ot)
    if ns not in newxy or os not in oldxy:
     external.append(dict(clone=clone,pin=pin,driver=ns,old_driver=os));continue
    b=dist(oldxy,os,original);c=dist(newxy,ns,clone)
    category='reset' if pin=='SR' else 'enable' if pin=='CE' else 'srl_address' if k=='EFX_SRL8' and pin=='A' else 'carry' if k=='EFX_ADD' and pin=='CI' else 'data'
    rows.append(dict(clone=clone,original=original,kind=k,pin=pin,category=category,driver=ns,old_driver=os,before_distance=b,after_distance=c,delta=c-b))
 groups={}
 for category in sorted({r['category'] for r in rows}):
  rr=[r for r in rows if r['category']==category]
  groups[category]=dict(edges=len(rr),before_mean=sum(r['before_distance'] for r in rr)/len(rr),after_mean=sum(r['after_distance'] for r in rr)/len(rr),max_after=max(r['after_distance'] for r in rr),worsened=sum(r['delta']>0 for r in rr))
 evidence=[BOARD/'result.json',TRANSFORM/'result.json',BOARD/'outflow/ti60f225_oob.map.v',PARENT/'outflow/ti60f225_oob.map.v',prior/'outflow/ti60f225_oob.out.placeloc',place/'outflow/ti60f225_oob.out.placeloc',Path(__file__)]
 result=dict(status='TARGETED_REPLICA_UPSTREAM_GEOMETRY_REVIEW',groups=groups,edges=rows,worst_transfers=sorted(rows,key=lambda r:-r['delta'])[:20],external_driver_edges=external,added_clock_loads=clock_loads,
  scope='Compare every non-clock replica input to the corresponding original input placement. Distances are estimates, not routed delays or dedicated clock/reset route checks. Original and duplicated downstream edges and whole-chip congestion are reviewed separately.',evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
 a.out.write_text(json.dumps(result,indent=2));print(json.dumps(groups));print(json.dumps(result['worst_transfers'][:5]))
if __name__=='__main__':main()
