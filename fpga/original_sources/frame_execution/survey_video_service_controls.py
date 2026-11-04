"""Read-only public control-net geometry review; not routed conflict attribution."""
import argparse,collections,json,pickle
from pathlib import Path
from run_units import HERE,ROOT,sha
from fixed_control_ddr_reuse_locations import locations

OUTPUTS={'EFX_LUT4':{'O'},'EFX_FF':{'Q'},'EFX_ADD':{'O','CO'},'EFX_SRL8':{'Q'}}
CLOCKS={'CLK','WCLK','RCLK','CLKA','CLKB'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
 p.add_argument('--physical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();physical=a.physical.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 mapped=physical/'outflow/ti60f225_oob.map.v';pos=physical/'outflow/ti60f225_oob.out.placeloc'
 assert sha(mapped)==sha(board/'outflow/ti60f225_oob.map.v')
 cache=board/'service-validation-graph.pkl';key,g=pickle.loads(cache.read_bytes())
 assert key==(sha(mapped),sha(HERE/'audit_continuous_vectors.py'),sha(HERE/'native_local_drivers.py'))
 xy=locations(pos);drivers={};loads=collections.defaultdict(list)
 for n,(k,pins) in g.items():
  for pin,ts in pins.items():
   if pin in OUTPUTS.get(k,set()):
    for t in ts:
     if t[0]=='net':drivers[t]=n
   elif pin not in CLOCKS:
    for t in ts:
     if t[0]=='net':loads[t].append((n,pin))
 rows=[]
 for t,driver in drivers.items():
  sinks=[(n,p) for n,p in loads[t] if n!=driver]
  if len(sinks)<100:continue
  points=[xy[n] for n,p in sinks if n in xy]+([xy[driver]] if driver in xy else [])
  assert points
  groups=collections.Counter(n.split('/')[0] if n.startswith('service_') else 'other_public' for n,p in sinks)
  rows.append(dict(net=t,driver=driver,fanout=len(sinks),distinct_cells=len({n for n,p in sinks}),
   pins=dict(collections.Counter(p for n,p in sinks)),sinks_by_group=dict(groups),
   bbox=[min(x[0] for x in points),min(x[1] for x in points),max(x[0] for x in points),max(x[1] for x in points)],
   unplaced_sink_cells=sorted({n for n,p in sinks if n not in xy}),service_driver=driver.startswith('service_')))
 rows.sort(key=lambda r:r['fanout'],reverse=True)
 result=dict(status='PUBLIC_CONTROL_NETWORK_GEOMETRY_REVIEW_NOT_ROUTED_CONFLICTS',
  physical=str(physical),map_sha256=sha(mapped),placement_sha256=sha(pos),rows=rows,
  cache_key=list(key),cache_sha256=sha(cache),checker_sha256=sha(__file__),
  scope='Known public LUT/FF/ADD/SRL drivers; explicit net bit references expanded, constants and clock pins excluded, resets/enables included. Groups are name prefixes, not retained/new equivalence classification. Bboxes are geometry, not track occupancy or shared-control compatibility proof.',
  hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(physical=str(physical),nets=len(rows),new_control=[r for r in rows if r['service_driver']]),ensure_ascii=False))

if __name__=='__main__':main()
