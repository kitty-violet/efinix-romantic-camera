"""Verify the factored control cone does not create asynchronous reset use."""
import collections,json,pickle
from pathlib import Path
from run_units import ROOT,EF,sha
from video_control85_physical import BOARD,audit

def trace(board):
 h,g,p,d,l=pickle.loads((board/'integration-parse.pkl').read_bytes());assert h==sha(board/'outflow/ti60f225_oob.map.v')
 q=[('net','service63_ddr_33',None),('net','control_remap_1152',None)];seen=set();ends=[];ops={'EFX_LUT4':('O',),'EFX_ADD':('O','CO')}
 while q:
  t=q.pop()
  if t in seen:continue
  seen.add(t)
  for n,pin in l.get(t,[]):
   k,ports=g[n]
   if k in ops:
    for op in ops[k]:q.extend(ports.get(op,[]))
   else:ends.append(dict(cell=n,kind=k,pin=pin,clock=ports.get('CLK'),parameters=p.get(n,{})))
 return h,seen,ends

def main():
 audit();old=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'
 h,signals,ends=trace(BOARD);oh,_,prior=trace(old)
 special=('SR','RST','RESET','CLK','CLKA','CLKB','WCLK','RCLK')
 rows=[r for r in ends if r['pin'] in special];previous=[r for r in prior if r['pin'] in special]
 def key(r):return r['cell'],r['pin']
 assert sorted(rows,key=key)==sorted(previous,key=key)
 assert len(rows)>0
 for r in rows:
  assert r['clock']==[('net','process_clk_150',None)]
  if r['kind']=='EFX_FF':assert r['pin']=='SR' and r['parameters']['SR_SYNC']==(1,1)
  else:assert r['kind']=='EFX_DSP48' and r['pin']=='RST' and r['parameters']['RST_SYNC']==(1,1)
 model=EF/'sim_models/verilog/efx_dsp48.v';assert '.SYNC(RST_SYNC)' in model.read_text()
 result=dict(status='CONTROL85_FACTORED_CONE_SYNCHRONOUS_RESET_SINKS_PASS_NOT_ROUTED',map_sha256=h,parent_map_sha256=oh,
  public_signals=len(signals),public_endpoints=len(ends),special_endpoints=rows,reset_sink_counts=dict(collections.Counter(r['kind'] for r in rows)),
  all_special_sinks_unchanged=True,asynchronous_reset_sinks=0,clock_sinks=0,
  evidence={str(f):sha(f) for f in (Path(__file__),model,BOARD/'control85-preplace-audit.json',BOARD/'outflow/ti60f225_oob.map.v',old/'outflow/ti60f225_oob.map.v')},
  limits=['Public combinational fanout analysis only. Actual setup/hold at these synchronous reset pins still needs routed STA.'],hardware_access=False)
 (BOARD/'control-reset-fanout-audit.json').write_text(json.dumps(result,indent=2));print(result['status'],result['reset_sink_counts'])
if __name__=='__main__':main()
