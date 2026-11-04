"""Bounded finite-capacity comparison; no timing claim or model modification."""
import argparse,itertools,json
from pathlib import Path
from fourway_frame_program import compile_tile,hw,compile_contract,load_contract
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
plan=compile_contract(load_contract());group=next(g for g in hw.build_tile_manifest(320)['groups'] if g['name']=='residual');rows=[]
patterns=[None]+[(d1,p1,d2,out) for d1 in (1,2) for p1 in (0,1,2) if p1!=d1 for d2 in (1,2) if d2!=p1 for out in (0,1,2) if out!=d2]
for width,lead,retain in itertools.product((96,80,72),(4,2,0),(False,True)):
 for pattern in patterns:
  tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1)) for i,x in enumerate(range(0,320,width))]
  try:
   parts=[compile_tile(plan,group,t,-1,4,plane_shift=True,prefetch_rows=lead,retain_commit=retain,compact_lifetime=True,plane_pattern=pattern) for t in tiles]
   row=dict(width=width,lead=lead,retain=retain,pattern=pattern,passed=True,peak=max(t['peak_live_padded_bytes'] for t in parts),high=[max(t['finite_high_water'][i] for t in parts) for i in range(3)])
   print(json.dumps(row),flush=True)
  except MemoryError as e:row=dict(width=width,lead=lead,retain=retain,pattern=pattern,passed=False,error=str(e))
  rows.append(row)
a.output.write_text(json.dumps(dict(status='CAPACITY_ONLY',rows=rows),indent=2));print('CAPACITY_PROBE_COMPLETE',flush=True)
