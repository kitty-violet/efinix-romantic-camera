"""Conservative read-only unused-public-logic survey on the actual mapped graph.

No source or map is edited. Top ports, assignments, nonprimitive instances,
opaque definitions, fixed anchors and explicit SDC names remain roots.
"""
import argparse,collections,json,pickle,re
from pathlib import Path
from run_units import HERE,ROOT,sha
from native_local_drivers import CELL,REF
from fixed_control_ddr_reuse_locations import locations

OUTPUTS={'EFX_LUT4':{'O'},'EFX_FF':{'Q'},'EFX_ADD':{'O','CO'},'EFX_SRL8':{'Q'}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
 p.add_argument('--physical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();physical=a.physical.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 mapped=board/'outflow/ti60f225_oob.map.v';text=mapped.read_text()
 cache=board/'service-validation-graph.pkl';key,g=pickle.loads(cache.read_bytes())
 assert key==(sha(mapped),sha(HERE/'audit_continuous_vectors.py'),sha(HERE/'native_local_drivers.py'))
 assert sha(physical/'outflow/ti60f225_oob.map.v')==sha(mapped)
 exterior=CELL.sub('',text)
 exterior=re.sub(r'^\s*defparam\s+[^;]*;','',exterior,flags=re.M)
 exterior=re.sub(r'^\s*wire\s+[^;]*;','',exterior,flags=re.M)
 # Comments may mention unused names. Keeping such names is conservative.
 roots={m['id'].lstrip('\\') for m in REF.finditer(exterior) if m['id']}
 sdc=(physical/'ti60f225_oob.sdc').read_text();anchors=locations(physical/'fixed_dsp.placeloc')
 xy=locations(physical/'outflow/ti60f225_oob.out.placeloc')
 outputs={};uses=collections.defaultdict(set)
 for n,(k,pins) in g.items():
  outputs[n]={t for p,ts in pins.items() if p in OUTPUTS.get(k,set()) for t in ts if t[0]=='net'}
  for p,ts in pins.items():
   if p not in OUTPUTS.get(k,set()):
    for t in ts:
     if t[0]=='net':uses[t].add(n)
 candidates={n for n,(k,pins) in g.items() if k in ('EFX_LUT4','EFX_FF') and outputs[n] and
             not n.startswith('protected_inst/') and n not in anchors and n not in sdc and
             not any(t[1] in roots for t in outputs[n])}
 removed=set();rounds=[]
 while True:
  new={n for n in candidates-removed if not any(uses[t]-removed for t in outputs[n])}
  if not new:break
  rounds.append(sorted(new));removed.update(new)
 # A dead state loop has fanout to itself and cannot be found by leaf pruning.
 # Treat every unknown/side-effecting primitive and exterior binding as live,
 # then trace ALL inputs (including clock/reset/enable) backwards.
 pure={'EFX_LUT4','EFX_FF','EFX_ADD','EFX_SRL8'}
 drivers=collections.defaultdict(set)
 for n,ts in outputs.items():
  for t in ts:drivers[t].add(n)
 live={n for n,(k,pins) in g.items() if k not in pure or n.startswith('protected_inst/') or
       n in anchors or n in sdc or any(t[1] in roots for t in outputs[n])}
 pending=list(live)
 while pending:
  n=pending.pop();k,pins=g[n]
  for pin,ts in pins.items():
   if pin in OUTPUTS.get(k,set()):continue
   for t in ts:
    for driver in drivers[t]-live:
     live.add(driver);pending.append(driver)
 unobservable=set(g)-live
 assert removed<=unobservable
 cone_rows=[dict(name=n,kind=g[n][0],placed=list(xy[n]) if n in xy else None) for n in sorted(unobservable)]
 warnings=sorted(set(re.findall(r'logical_block (.+?) #\d+ has no fanout',(physical/'outflow/ti60f225_oob.place.out').read_text())))
 rows=[dict(name=n,kind=g[n][0],placed=list(xy[n]) if n in xy else None,outputs=sorted(outputs[n],key=str)) for n in sorted(removed)]
 result=dict(status='UNUSED_PUBLIC_LOGIC_CANDIDATES_ONLY_NO_TRANSFORMATION',map_sha256=sha(mapped),
   physical=str(physical),placement_sha256=sha(physical/'outflow/ti60f225_oob.out.placeloc'),
   direct_tool_warnings=len(warnings),warning_candidates=sorted(set(warnings)&removed),warning_not_proved=sorted(set(warnings)-removed),
   candidates=len(rows),kinds=dict(collections.Counter(r['kind'] for r in rows)),rounds=rounds,rows=rows,
   unobservable_cone_candidates=len(cone_rows),unobservable_cone_kinds=dict(collections.Counter(r['kind'] for r in cone_rows)),unobservable_cone_rows=cone_rows,
   cache_key=list(key),checker_sha256=sha(__file__),constraints_sha256=sha(physical/'ti60f225_oob.sdc'),
   scope='Public LUT/FF leaf reachability only; unknown primitive ports and all exterior identifiers conservatively live. Actual XLR savings and a source-to-map removal proof require a separate candidate. Current routed candidate remains unchanged.',
   hardware_access=False,source_edited=False,board_release=False)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','direct_tool_warnings','candidates','kinds','warning_not_proved','unobservable_cone_candidates','unobservable_cone_kinds')}))

if __name__=='__main__':main()
