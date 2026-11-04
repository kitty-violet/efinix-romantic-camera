"""Whole-netlist load transfer audit before placement; no geometry claims."""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,sha
from camera_joint_board import PARENT,TRANSFORM,transform_check
from camera_joint_native import OUT
from native_local_drivers import graph
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 proof=transform_check();old=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());new=graph((TRANSFORM/'native_source.v').read_text())
 def loads(g):
  r=collections.defaultdict(set)
  for n,(k,pins) in g.items():
   for pin,ts in pins.items():
    if pin not in OUT.get(k,[]):
     for t in ts:
      if t[0]!='literal':r[t].add((n,pin))
  return r
 before,after=loads(old),loads(new)
 def row(t):
  pins=sorted({p for n,p in after[t]});kind='clock' if any('CLK' in p and 'CLKE' not in p for p in pins) else 'reset' if any(p in ('SR','RST','RESET') for p in pins) else 'data_enable'
  return dict(net=t,before=len(before[t]),after=len(after[t]),delta=len(after[t])-len(before[t]),category=kind,pins=pins)
 inc=sorted([row(t) for t in before if len(after[t])>len(before[t])],key=lambda r:-r['delta'])
 split=[]
 for r in proof['replicas']:
  t=tuple(r['net']);outputs=[t]+[tuple(c['output']) for c in r['clones']]
  split.append(dict(driver=r['driver'],kind=r['kind'],before=len(before[t]),groups=len(outputs),max_after=max(len(after[t]) for t in outputs),total_after=sum(len(after[t]) for t in outputs)))
 # Explicitly prove no new asynchronous sequential replicas or other clocks.
 from probe_full_native_split_map import params
 ps=params((TRANSFORM/'native_source.v').read_text())
 sequential=[(n,k,p) for n,(k,p) in new.items() if n.startswith('JOINT_LOCAL_') and k in ('EFX_FF','EFX_SRL8')]
 assert all(p['CLK']==[('net','process_clk_150',None)] for n,k,p in sequential)
 assert all(ps[n]['SR_SYNC']==(1,1) for n,k,p in sequential if k=='EFX_FF')
 result=dict(status='JOINT_WHOLE_NETLIST_LOAD_TRANSFER_REVIEW',split_drivers=split,top_increases=inc[:40],
  changed_clock_reset_loads=[r for r in inc if r['category']!='data_enable'],synchronous_replicas=len(sequential),
  risks=['Placement must absorb the added upstream data/enable/reset loads; lower output fanout does not by itself prove timing or routability.','Dedicated ADD cascade links must not have syn_keep attributes.'],
  evidence={str(f):sha(f) for f in (TRANSFORM/'result.json',TRANSFORM/'native_source.v',PARENT/'outflow/ti60f225_oob.map.v',Path(__file__))},hardware_access=False,board_release=False)
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','changed_clock_reset_loads','synchronous_replicas')}));print(json.dumps(inc[:6]));print(json.dumps(sorted(split,key=lambda r:-r['before'])[:8]))
if __name__=='__main__':main()
