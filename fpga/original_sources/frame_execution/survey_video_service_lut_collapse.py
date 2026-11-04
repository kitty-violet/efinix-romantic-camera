"""Find exact single-consumer LUT compositions with at most four inputs.

This is a read-only survey. Exterior/protected/constraint nets are boundaries;
no state, clocks, timing exceptions, or multiple-consumer fanout is changed.
"""
import argparse,collections,itertools,json,re,time
from pathlib import Path
from run_units import ROOT,sha
from video_service_graph import load
from native_local_drivers import CELL,REF
from camera_joint_native import OUT,used_inputs

ZERO=('literal',"1'b0",None)
def value(mask,ins,values):
 idx=0
 for i,t in enumerate(ins):
  if t[0]=='literal':
   assert t[1] in ("1'b0","1'b1"),t
   b=int(t[1][-1])
  else:b=values[t]
  idx|=b<<i
 return mask>>idx&1

def survey(native,physical):
 start=time.monotonic();proof=json.loads((native/'native-graph-audit.json').read_text())
 assert proof['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
 for f,h in proof['inputs'].items():assert sha(f)==h,f
 graft=json.loads((native/'graft.json').read_text());D=load()
 removed=set(graft['removed']);changes={}
 for r in graft['changed_loads']:changes.setdefault((r['cell'],r['pin']),{})[tuple(r['from_token'])]=tuple(r['to_token'])
 g={n:(k,{p:[changes.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in D['g'].items() if n not in removed}
 assert len(g)==proof['retained_primitives']
 ps=D['ps'];s=(native/'native_source.v').read_text();outer=CELL.sub('',s)
 outer=re.sub(r'^\s*defparam\s+[^;]*;','',outer,flags=re.M)
 outer=re.sub(r'^\s*wire\s+[^;]*;','',outer,flags=re.M)
 roots={m['id'].lstrip('\\') for m in REF.finditer(outer) if m['id']}
 sdc=(physical/'ti60f225_oob.sdc').read_text();loads=collections.defaultdict(set);driver={}
 for n,(k,pins) in g.items():
  for p,ts in pins.items():
   for t in ts:
    if p in OUT[k]:driver[t]=n
    else:loads[t].add((n,p))
 rows=[];used=set()
 for child,(k,pins) in sorted(g.items()):
  if k!='EFX_LUT4' or child in used or child in sdc:continue
  out=pins['O'][0]
  if any(p in ('CLK','WCLK','RCLK','CLKA','CLKB') for n,p in loads[out]):continue
  child_used={t for ts in used_inputs(k,pins,ps[child]).values() for t in ts if t[0]!='literal'}
  for t in sorted(child_used,key=str):
   parent=driver.get(t)
   if parent is None or parent in used or g[parent][0]!='EFX_LUT4':continue
   if parent in sdc or t[1] in sdc or t[1] in roots:continue
   if {n for n,p in loads[t]}!={child}:continue
   pp=g[parent][1]
   boundary=sorted((child_used-{t})|{v for ts in used_inputs('EFX_LUT4',pp,ps[parent]).values() for v in ts if v[0]!='literal'},key=str)
   if len(boundary)>4 or out in boundary:continue
   pi=[pp.get('I'+str(i),[ZERO])[0] for i in range(4)]
   ci=[pins.get('I'+str(i),[ZERO])[0] for i in range(4)]
   # Unused physical inputs need values but cannot affect the truth table.
   pi=[v if v in boundary or v[0]=='literal' else ZERO for v in pi]
   ci=[v if v in boundary or v==t or v[0]=='literal' else ZERO for v in ci]
   table=[]
   for x in range(1<<len(boundary)):
    vals={v:(x>>i)&1 for i,v in enumerate(boundary)}
    vals[t]=value(ps[parent]['LUTMASK'][1],pi,vals)
    table.append(value(ps[child]['LUTMASK'][1],ci,vals))
   mask=sum(table[x&((1<<len(boundary))-1)]<<x for x in range(16))
   rows.append(dict(parent=parent,child=child,removed_net=t,output=out,inputs=boundary,lutmask=mask,
                    original_parent_mask=ps[parent]['LUTMASK'][1],original_child_mask=ps[child]['LUTMASK'][1],truth=table))
   used.update((parent,child));break
 return dict(status='SINGLE_CONSUMER_FOUR_INPUT_LUT_COMPOSITION_SURVEY',native=str(native),
  native_sha256=sha(native/'native_source.v'),native_audit_sha256=sha(native/'native-graph-audit.json'),
  sdc_sha256=sha(physical/'ti60f225_oob.sdc'),source_checker_sha256=sha(__file__),
  candidates=len(rows),truth_cases=sum(len(r['truth']) for r in rows),rows=rows,
  seconds=time.monotonic()-start,hardware_access=False,source_edited=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--physical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 r=survey(a.native.resolve(),a.physical.resolve());out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
