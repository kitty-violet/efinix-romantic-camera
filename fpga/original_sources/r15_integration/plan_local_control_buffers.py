"""Plan bounded fanout branches using observed geometry, without changing RTL.

Two existing combinational buffer nets are candidates, not proven failed nets.
Clock, asynchronous reset and DSP cascade nets are not changed by this plan.
No placement is fixed: coordinates only choose groups of connected consumers.
"""
import collections,json,re
from audit_production_map import ROOT,PARENT,sha,parse,topology
from place_production_design import locations

def main():
 source=ROOT/'production/native_sapphire_control.v'
 s=parse(source);g,dr,ld=topology(s)
 base=ROOT/'build/production-r13'
 loc=locations(base/'outflow/ti60f225_oob.out.placeloc')
 oldloc=locations(PARENT/'outflow/ti60f225_oob.out.placeloc')
 groups=[]
 for label,net,bit in [('mode','lane_depthwise',23),('enable','pair_advance',7)]:
  token=('net','s4r1/core/dut/engine/mac/arithmetic/'+net,bit)
  name,pin,index=dr[token];kind,ports=g[name]
  mask=s[2][name]['LUTMASK'];value=mask[1] if isinstance(mask,(tuple,list)) else int(mask)
  assert kind=='EFX_LUT4' and pin=='O' and value==43690,(name,kind,pin,mask)
  consumers=collections.defaultdict(list)
  for n,p,i in sorted(ld[token]):
   assert p not in ('CLK','RCLK','WCLK','SR','RST','CASCIN'),(n,p)
   assert n in loc or n in oldloc,n
   consumers[n].append((p,i))
  points={n:(loc[n] if n in loc else oldloc[n]) for n in consumers}
  def split(names):
   weight=sum(len(consumers[n]) for n in names)
   if weight<=128:return [names]
   ranges=[max(points[n][a] for n in names)-min(points[n][a] for n in names) for a in (0,1)]
   axis=max(range(2),key=lambda a:ranges[a]);ordered=sorted(names,key=lambda n:(points[n][axis],points[n][1-axis],n))
   total=0;middle=1
   for i,n in enumerate(ordered[:-1]):
    total+=len(consumers[n]);middle=i+1
    if total>=weight/2:break
   return split(ordered[:middle])+split(ordered[middle:])
  branches=[]
  for i,names in enumerate(split(sorted(consumers))):
   rows=[dict(cell=n,pins=consumers[n],site=points[n],geometry='r13' if n in loc else 'accepted_native8h_fallback') for n in names]
   fanout=sum(len(r['pins']) for r in rows)
   branches.append(dict(name=f'control_local_{label}_{i}',fanout=fanout,consumers=rows))
  groups.append(dict(label=label,net=token,original_driver=name,original_input=ports['I0'],
   total_load_pins=len(ld[token]),branches=branches))
 result=dict(status='CONTROL_BUFFER_TOPOLOGY_PLAN_NOT_IMPLEMENTED',
  source=str(source),source_sha256=sha(source),geometry_sha256=sha(base/'outflow/ti60f225_oob.out.placeloc'),
  fallback_geometry_sha256=sha(PARENT/'outflow/ti60f225_oob.out.placeloc'),groups=groups,
  extra_LUT_estimate=sum(len(x['branches'])-1 for x in groups),
  placement_locks=0,new_pipeline_stages=0,clock_reset_cascade_changes=0,
  scope='Pure combinational duplication candidate; no full-system capacity/congestion/timing claim',hardware_access=False)
 (ROOT/'evidence/local-control-buffer-plan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(result['status'],'extra LUT estimate',result['extra_LUT_estimate'])
 for group in groups:print(group['label'],group['total_load_pins'],'->',[x['fanout'] for x in group['branches']])

if __name__=='__main__':main()
